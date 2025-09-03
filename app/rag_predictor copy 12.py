from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from pathlib import Path
from typing import List, Any, Optional, Set
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiofiles
import os
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XMLRAGPredictor:
    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL = "llama3:latest"
    
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    BATCH_SIZE = 8
    MAX_WORKERS = 2

    def __init__(self, persist_dir: str = "vectorstore") -> None:
        """Initialize the RAG predictor with vector store and LLM."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self._init_components(persist_dir)
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        self._semaphore = asyncio.Semaphore(self.MAX_WORKERS)
        self._prediction_cache = {}

    def _init_components(self, persist_dir: str) -> None:
        """Initialize model components with error handling."""
        try:
            os.makedirs(persist_dir, exist_ok=True)
            
            self.embedding = OllamaEmbeddings(
                model=self.EMBEDDING_MODEL
            )
            
            # Configure LLM with minimal settings
            self.llm = Ollama(
                model=self.LLM_MODEL,
                temperature=0.1  # Lower temperature for more focused responses
            )
            
            # Initialize Chroma client with optimized settings
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            
            # Initialize Chroma vector store
            self.vectorstore = Chroma(
                client=client,
                collection_name="xml_rag_collection",
                embedding_function=self.embedding,
                persist_directory=persist_dir
            )
            
            # Configure retriever with optimized settings
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 2,  # Reduce number of retrieved documents
                }
            )
            
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True
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
            
            async with aiofiles.open(diffs_file, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
            
            docs = splitter.split_text(content)
            
            if not docs:
                logger.warning("No documents were generated from the diff file")
                return

            tasks = []
            for batch in self._batch_generator(docs, self.BATCH_SIZE):
                task = asyncio.create_task(self._process_batch(batch))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.vectorstore.persist
            )
            
            logger.info(f"Training completed. Processed {len(docs)} documents.")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def _get_cached_prediction(self, content_hash: int) -> Optional[str]:
        """Get cached prediction if exists."""
        return self._prediction_cache.get(content_hash)

    def _cache_prediction(self, content_hash: int, prediction: str) -> None:
        """Cache a prediction."""
        self._prediction_cache[content_hash] = prediction
        if len(self._prediction_cache) > 1000:
            self._prediction_cache.pop(next(iter(self._prediction_cache)))

    async def predict_changes(self, v1_content: str, max_predictions: int = 5, timeout: int = 120) -> str:
        """
        Predict potential changes for the given XML content with caching and optimizations.
        
        Args:
            v1_content: The XML content to predict changes for
            max_predictions: Maximum number of predictions to return (reduced default)
            timeout: Maximum time in seconds to wait for prediction (increased default to 120s)
            
        Returns:
            str: Predicted changes in natural language
        """
        if not v1_content.strip():
            return ""
            
        content_hash = hash(v1_content)
        
        cached = self._get_cached_prediction(content_hash)
        if cached is not None:
            return cached
        
        try:
            if len(v1_content) > 5000: 
                return await self._predict_changes_in_chunks(v1_content, max_predictions, timeout)
            
            prompt = self._generate_optimized_prompt(v1_content, max_predictions)
            
            loop = asyncio.get_event_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        lambda: self.qa.invoke({"query": prompt})
                    ),
                    timeout=timeout
                )
                
                if isinstance(result, dict) and 'result' in result:
                    result = result['result']
                
                self._cache_prediction(content_hash, result)
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Prediction timed out after {timeout} seconds")
                return ""
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return ""

    async def _predict_changes_in_chunks(self, content: str, max_predictions: int, timeout: int) -> str:
        """Process large content in chunks."""
        try:
            from lxml import etree
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(f"<root>{content}</root>".encode(), parser=parser)
            chunks = [etree.tostring(elem, encoding='unicode') for elem in root]
            if len(chunks) > 1:
                pass
            else:
                raise Exception("Couldn't split by XML elements")
        except Exception:
            chunk_size = 3000 
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        results = []
        chunk_timeout = max(30, timeout // max(1, len(chunks)))
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                chunk_result = await self.predict_changes(
                    chunk, 
                    max_predictions=max(1, max_predictions // len(chunks)),
                    timeout=chunk_timeout,
                )
                if chunk_result and chunk_result.strip():
                    results.append(chunk_result.strip())
            except Exception as e:
                logger.warning(f"Error processing chunk {i+1}: {str(e)}")
        
        unique_results = []
        seen = set()
        for res in results:
            if res and res not in seen:
                seen.add(res)
                unique_results.append(res)
                if len(unique_results) >= max_predictions:
                    break
        
        return "\n".join(unique_results)

    def _generate_optimized_prompt(self, v1_content: str, max_predictions: int) -> str:
        """Generate a more efficient prompt for change prediction."""
        truncated_content = v1_content[:6000]
        
        return f"""
Analyze this XML and list up to {max_predictions} most likely changes.
Focus on:
1. Structural changes (add/remove elements)
2. Attribute modifications
3. Content updates

XML (truncated):
{truncated_content}

Format each change concisely in one line:
- [CHANGE_TYPE] [XPATH] | Current: [VALUE] | Suggested: [NEW_VALUE]
""".strip()

    async def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        if hasattr(self, 'vectorstore'):
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.vectorstore.persist
            )

    def __del__(self):
        """Ensure resources are properly cleaned up."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        if hasattr(self, 'vectorstore'):
            self.vectorstore = None
        if hasattr(self, 'llm'):
            self.llm = None
        if hasattr(self, 'embedding'):
            self.embedding = None
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