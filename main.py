from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load environment variables
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


# 2. Define Chroma database path
db_path = BASE_DIR / "chroma_db"


# 3. Initialize embeddings model
print("Loading embedding model...")

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Connect to existing Chroma database
print("Connecting to Chroma database...")

vectorstore = Chroma(
    persist_directory=str(db_path),
    embedding_function=embeddings_model,
    collection_name="research",
)

# Check stored documents
total_chunks = vectorstore._collection.count()

print("Total stored chunks:", total_chunks)

if total_chunks == 0:
    raise ValueError("Chroma database is empty! Run create_database.py first.")


# 5. Initialize retriever
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)


# 6. Initialize Groq chat model
print("Loading Groq model...")

model = init_chat_model("groq:openai/gpt-oss-120b")


# 7. Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

            Answer the user's question using ONLY the provided context.

            If the answer is not present in the context,
            say: "I could not find the answer in the document."

            Do not make up information.
            """,
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}""",
        ),
    ]
)


# 8. Start RAG application
print("\nRAG system created successfully!")
print("Ask questions about your PDF.")
print("Press 0 to exit.\n")


while True:

    query = input("You: ").strip()

    if query == "0":
        print("Exiting RAG system...")
        break

    if not query:
        continue

    # 9. Retrieve relevant document chunks
    docs = retriever.invoke(query)

    print("\nRetrieved documents:", len(docs))

    # Debugging: Display retrieved chunks
    for i, doc in enumerate(docs):
        print(f"\nChunk {i + 1}:")
        print(doc.page_content[:300])
        print("Metadata:", doc.metadata)

    # 10. Check retrieval results
    if not docs or not any(doc.page_content.strip() for doc in docs):
        print("\nAI: I could not find the answer in the document.\n")
        continue

    # 11. Combine retrieved chunks into context
    context = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())

    # 12. Create final prompt
    final_prompt = prompt.invoke({"context": context, "question": query})

    # 13. Generate answer
    response = model.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")
