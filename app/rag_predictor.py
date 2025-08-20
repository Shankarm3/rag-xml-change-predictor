from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import aiofiles
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XMLRAGPredictor:
    # Model configuration
    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL = "llama3:latest"
    
    # Performance configuration
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 100
    BATCH_SIZE = 32  # Optimal batch size for vector operations
    MAX_WORKERS = 4  # Number of parallel workers for CPU-bound tasks

    def __init__(self, persist_dir: str = "vectorstore") -> None:
        """Initialize the RAG predictor with vector store and LLM."""
        self._init_components(persist_dir)
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        self._semaphore = asyncio.Semaphore(self.MAX_WORKERS)

    def _init_components(self, persist_dir: str) -> None:
        """Initialize model components with error handling."""
        try:
            self.embedding = OllamaEmbeddings(model=self.EMBEDDING_MODEL)
            self.llm = Ollama(model=self.LLM_MODEL)
            self.vectorstore = Chroma(
                persist_directory=persist_dir, 
                embedding_function=self.embedding,
                collection_metadata={"hnsw:space": "cosine"}  # Optimize for similarity search
            )
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm, 
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
            )
            logger.info("RAG predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG predictor: {str(e)}")
            raise

    async def _process_batch(self, batch: List[str]) -> None:
        """Process a batch of documents asynchronously."""
        async with self._semaphore:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.vectorstore.add_texts(batch)
                )
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                raise

    def _batch_generator(self, items: List[Any], batch_size: int):
        """Generate batches from a list."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    async def train_from_diffs(self, diffs_file: str, 
                             chunk_size: Optional[int] = None, 
                             chunk_overlap: Optional[int] = None) -> None:
        """
        Train the model from a file containing XML diffs with parallel processing.
        
        Args:
            diffs_file: Path to the file containing XML diffs
            chunk_size: Size of text chunks (default: DEFAULT_CHUNK_SIZE)
            chunk_overlap: Overlap between chunks (default: DEFAULT_CHUNK_OVERLAP)
        """
        if not Path(diffs_file).exists():
            raise FileNotFoundError(f"Diff file not found: {diffs_file}")

        chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        chunk_overlap = chunk_overlap or self.DEFAULT_CHUNK_OVERLAP
        
        try:
            logger.info(f"Processing diffs from {diffs_file}")
            
            # Use aiofiles for async file I/O
            async with aiofiles.open(diffs_file, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Process text in chunks in parallel
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
            
            # Split documents
            docs = splitter.split_text(content)
            
            if not docs:
                logger.warning("No documents were generated from the diff file")
                return

            # Process in parallel batches
            tasks = []
            for batch in self._batch_generator(docs, self.BATCH_SIZE):
                task = asyncio.create_task(self._process_batch(batch))
                tasks.append(task)
            
            # Wait for all batches to complete
            await asyncio.gather(*tasks)
            
            # Persist once at the end
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.vectorstore.persist
            )
            
            logger.info(f"Training completed. Processed {len(docs)} documents.")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    @lru_cache(maxsize=128)
    def _get_cached_prediction(self, content_hash: int, v1_content: str) -> str:
        """Cache predictions to avoid redundant processing."""
        return self.qa.run(v1_content)

    async def predict_changes(self, v1_content: str) -> str:
        """
        Predict potential changes for the given XML content with caching.
        
        Args:
            v1_content: The XML content to predict changes for
            
        Returns:
            str: Predicted changes in natural language
        """
        if not v1_content.strip():
            raise ValueError("Input content cannot be empty")
            
        # Create a hash for caching
        content_hash = hash(v1_content)
        
        try:
            logger.info("Generating change predictions")
            
            # Use cached result if available
            if content_hash in self._get_cached_prediction.cache_info().hits:
                logger.debug("Using cached prediction")
                return self._get_cached_prediction(content_hash, v1_content)
            
            prompt = self._generate_prompt(v1_content)
            
            # Run prediction in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.qa.run(prompt)
            )
            
            # Cache the result
            self._get_cached_prediction.cache_clear()  # Prevent cache from growing too large
            self._get_cached_prediction(content_hash, v1_content)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def _generate_prompt(self, v1_content: str) -> str:
        """Generate the prompt for change prediction."""
        return f"""
You are an XML change prediction assistant. Given the following XML content (v1), 
suggest the most likely changes that would be made to transform it to v2.

<v1>
{v1_content}
</v1>

Consider the following aspects:
1. Common structural changes (adding/removing elements)
2. Common attribute modifications
3. Common content updates
4. Formatting changes

Format your response as a list of specific, actionable change suggestions.
Each suggestion should start with a bullet point (-) and be as specific as possible.
"""

    async def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        if hasattr(self, 'vectorstore'):
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.vectorstore.persist
            )

    def __del__(self):
        """Ensure resources are cleaned up on object destruction."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

# Example async usage:
# async def main():
#     predictor = XMLRAGPredictor()
#     try:
#         await predictor.train_from_diffs("path/to/diffs.jsonl")
#         result = await predictor.predict_changes("<xml>...</xml>")
#         print(result)
#     finally:
#         await predictor.close()
# 
# asyncio.run(main())