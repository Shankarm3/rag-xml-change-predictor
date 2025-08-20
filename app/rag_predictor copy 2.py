from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

class XMLRAGPredictor:
    def __init__(self, persist_dir="vectorstore"):
        self.embedding = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = Ollama(model="llama3:latest")
        self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=self.embedding)
        self.qa = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.vectorstore.as_retriever())

    def train_from_diffs(self, diffs_file):
        docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        with open(diffs_file, 'r', encoding='utf-8') as f:
            for line in f:
                docs.extend(splitter.split_text(line))
        self.vectorstore.add_texts(docs)
        self.vectorstore.persist()

    def predict_changes(self, v1_content):
        prompt = f"""
Given the following XML content (v1), and based on prior editing patterns, what changes might be made to transform it to v2?

<v1>
{v1_content}
</v1>

Only suggest likely tag, structure, or content edits.
"""
        return self.qa.run(prompt)