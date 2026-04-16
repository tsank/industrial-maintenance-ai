import os
import sys
# from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env', override=True)

api_key = os.getenv("OPENAI_API_KEY", "MISSING")
print(f"DEBUG key length: {len(api_key)}")
print(f"DEBUG key start: {api_key[:15]}")
print(f"DEBUG key end: {api_key[-10:]}")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
COLLECTION_NAME = "maintenance_manuals"
DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "docs")

def get_chroma_client():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=int(CHROMA_PORT),
    )

def get_vectorstore(client):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

def load_documents(docs_path):
    from pdf_loader import load_pdf_robust

    documents = []
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in " + docs_path)
        sys.exit(1)

    for filename in pdf_files:
        filepath = os.path.join(docs_path, filename)
        print("Loading: " + filename)
        docs = load_pdf_robust(filepath, filename)
        documents.extend(docs)
        print("  Total pages extracted: " + str(len(docs)))

    print("Total pages across all documents: " + str(len(documents)))
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print("Total chunks created: " + str(len(chunks)))
    return chunks

def ingest():
    print("=" * 50)
    print("Equipment Manual Ingestion Pipeline")
    print("=" * 50)
    print("\n[1/3] Loading PDF documents...")
    documents = load_documents(DOCS_PATH)
    print("Total pages loaded: " + str(len(documents)))
    print("\n[2/3] Chunking documents...")
    chunks = chunk_documents(documents)
    print("\n[3/3] Embedding and storing in ChromaDB...")
    client = get_chroma_client()
    vectorstore = get_vectorstore(client)
    batch_size = 50
    total = len(chunks)
    num_batches = (total + batch_size - 1) // batch_size
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        batch_num = i // batch_size + 1
        done = min(i + batch_size, total)
        print("Batch " + str(batch_num) +  "/" + str(num_batches) + " - " + str(done) + "/" + str(total) + " chunks ingested")
    print("Successfully ingested " + str(len(chunks)) + " chunks")
    print("Collection: " + COLLECTION_NAME)
    print("ChromaDB: " + CHROMA_HOST + ":" + CHROMA_PORT)
    print("\nIngestion complete.")

if __name__ == "__main__":
    ingest()