# 1. Load environment variables
from dotenv import load_dotenv
from pathlib import Path

# 2. Import LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 3. Get project directory
BASE_DIR = Path(__file__).parent

# 4. Define PDF and database paths
pdf_path = BASE_DIR / "research.pdf"
db_path = BASE_DIR / "chroma_db"

# 5. Load PDF
print("Loading PDF...")

pdf_loader = PyPDFLoader(str(pdf_path))
docs = pdf_loader.load()

print("Total pages loaded:", len(docs))

# 6. Split PDF into chunks
print("Splitting documents...")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = splitter.split_documents(docs)

print("Total chunks created:", len(chunks))

# 7. Initialize embedding model
print("Loading embedding model...")

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 8. Create Chroma vector database
print("Creating Chroma database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=str(db_path),
    collection_name="research",
)

# 9. Verify database
print("\nDatabase created successfully!")
print("Total stored chunks:", vectorstore._collection.count())
print("Database location:", db_path)
