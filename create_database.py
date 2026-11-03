import os
import shutil
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# Load .env from the project directory
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# File paths
pdf_path = BASE_DIR / "research.pdf"
db_path = BASE_DIR / "chroma_db"

start_time = time.time()

print("\n========== CONFIGURATION ==========")
print("ENV file exists:", env_path.exists())
print("Google API key loaded:", bool(os.getenv("GOOGLE_API_KEY")))


# =====================================================
# 2. GEMINI HOSTED EMBEDDINGS
# =====================================================


class GeminiRAGEmbeddings(Embeddings):
    """
    Google Gemini hosted embeddings.

    Document embeddings use RETRIEVAL_DOCUMENT.
    Query embeddings use RETRIEVAL_QUERY.
    """

    def __init__(self):

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. "
                "Check your .env file in the project directory."
            )

        self.document_embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            google_api_key=api_key,
        )

        self.query_embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            task_type="RETRIEVAL_QUERY",
            google_api_key=api_key,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.document_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.query_embeddings.embed_query(text)


# =====================================================
# 3. CHECK PDF
# =====================================================

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")


# =====================================================
# 4. LOAD PDF
# =====================================================

print("\n[1/4] Loading PDF...")

step_start = time.time()

docs = PyPDFLoader(str(pdf_path)).load()

if not docs:
    raise ValueError("No pages found in PDF.")

print(f"Pages loaded: {len(docs)}")
print(f"Time: {time.time() - step_start:.2f} seconds")


# =====================================================
# 5. SPLIT PDF INTO CHUNKS
# =====================================================

print("\n[2/4] Splitting PDF...")

step_start = time.time()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

chunks = splitter.split_documents(docs)

if not chunks:
    raise ValueError("No text chunks created.")

print(f"Chunks created: {len(chunks)}")
print(f"Time: {time.time() - step_start:.2f} seconds")


# =====================================================
# 6. INITIALIZE GEMINI EMBEDDINGS
# =====================================================

print("\n[3/4] Initializing Gemini embeddings...")

step_start = time.time()

embeddings_model = GeminiRAGEmbeddings()

print("Gemini embeddings initialized successfully.")
print(f"Time: {time.time() - step_start:.2f} seconds")


# =====================================================
# 7. CREATE CHROMA DATABASE
# =====================================================

print("\n[4/4] Creating Chroma database...")

step_start = time.time()

# Remove the previous database if it exists
if db_path.exists():
    shutil.rmtree(db_path)
    print("Previous Chroma database removed.")

# Create a new database using Gemini embeddings
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=str(db_path),
    collection_name="research",
)

print(f"Stored chunks: {vectorstore._collection.count()}")
print(f"Database location: {db_path}")
print(f"Time: {time.time() - step_start:.2f} seconds")


# =====================================================
# 8. FINAL RESULT
# =====================================================

print("\n====================================")
print("DATABASE CREATED SUCCESSFULLY!")
print("Embedding model: Gemini embedding-001")
print(f"Total chunks: {len(chunks)}")
print(f"Total time: {time.time() - start_time:.2f} seconds")
print("====================================")
