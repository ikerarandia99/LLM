# ingest.py
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import *

DOCS_PATH = os.path.join(BASE_DIR, "docs")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

def load_documents():
    """
    Load all .txt files from /docs.
    """
    loader = DirectoryLoader(DOCS_PATH, glob="*.txt", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into smaller chunks for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_faiss_index(documents):
    """
    Create FAISS index using HuggingFace embeddings.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore

def save_faiss_index(vectorstore, path=INDEX_PATH):
    """
    Save FAISS index to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vectorstore.save_local(path)
    print(f"✅ FAISS index saved at: {path}")

if __name__ == "__main__":
    print("📂 Loading documents...")
    raw_docs = load_documents()

    print("✂️ Splitting documents...")
    docs = split_documents(raw_docs)

    print("🧠 Creating embeddings and FAISS index...")
    vectorstore = create_faiss_index(docs)

    print("💾 Saving index...")
    save_faiss_index(vectorstore)

    print("🚀 Ingestion completed successfully.")
