import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(page_title="AskMyPDF", page_icon="📚", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from the project directory.
# On Render, environment variables are supplied by Render itself.
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)


# ==================================================
# 2. GEMINI HOSTED EMBEDDINGS
# ==================================================


class GeminiRAGEmbeddings(Embeddings):
    """
    Google Gemini hosted embeddings.

    Documents use RETRIEVAL_DOCUMENT.
    Questions use RETRIEVAL_QUERY.

    No local Hugging Face or PyTorch model is loaded.
    """

    def __init__(self):

        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is missing. Add it to your "
                ".env file and Render environment variables."
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

    def embed_documents(self, texts):

        return self.document_embeddings.embed_documents(texts)

    def embed_query(self, text):

        return self.query_embeddings.embed_query(text)


@st.cache_resource
def load_embeddings():

    return GeminiRAGEmbeddings()


# ==================================================
# 3. GROQ LLM
# ==================================================


@st.cache_resource
def load_llm():

    from langchain_groq import ChatGroq

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0,
    )


# ==================================================
# 4. PROMPTS
# ==================================================

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an intelligent research assistant.

            Answer questions using ONLY the supplied document context.

            Rules:
            1. Give accurate and useful answers.
            2. Explain difficult concepts in simple language.
            3. Use conversation history for follow-up questions.
            4. Do not invent information.
            5. If the answer is not present in the context,
               say that you could not find it in the documents.
            6. Mention source filenames and page numbers when available.
            """,
        ),
        (
            "human",
            """
            Previous conversation:
            {chat_history}

            Retrieved document context:
            {context}

            Current question:
            {question}
            """,
        ),
    ]
)


map_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are summarizing a section of a research document.

            Create a concise but informative summary.
            Preserve important facts, methods, findings,
            numbers and technical terms.

            Do not add information that is not in the section.
            """,
        ),
        (
            "human",
            """
            Summarize this section:

            {text}
            """,
        ),
    ]
)


reduce_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are creating a comprehensive research paper summary.

            Combine the supplied section summaries into one coherent summary.

            Include:
            1. Research background
            2. Main objective
            3. Methodology
            4. Important findings
            5. Conclusions
            6. Limitations, if mentioned

            Do not invent information.
            Preserve important technical details.
            """,
        ),
        (
            "human",
            """
            Section summaries:

            {summaries}
            """,
        ),
    ]
)


# ==================================================
# 5. SESSION STATE
# ==================================================

defaults = {
    "messages": [],
    "documents": {},
    "vectorstore": None,
    "db_path": None,
    "selected_documents": [],
    "chat_document_key": None,
    "last_summary": None,
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# 6. INITIALIZE CHROMA DATABASE
# ==================================================


def initialize_database():

    if st.session_state.vectorstore is not None:
        return st.session_state.vectorstore

    from langchain_chroma import Chroma

    db_path = Path(tempfile.gettempdir()) / f"askmypdf_{uuid.uuid4().hex}"

    embeddings = load_embeddings()

    vectorstore = Chroma(
        collection_name="research_documents",
        embedding_function=embeddings,
        persist_directory=str(db_path),
    )

    st.session_state.db_path = str(db_path)
    st.session_state.vectorstore = vectorstore

    return vectorstore


# ==================================================
# 7. PROCESS PDF DOCUMENTS
# ==================================================


def process_documents(uploaded_files):

    from langchain_community.document_loaders import PyPDFLoader

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    vectorstore = initialize_database()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    processed = []

    for uploaded_file in uploaded_files:

        source_id = uuid.uuid4().hex

        temp_path = None

        try:

            # Save uploaded PDF temporarily.
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:

                temp_file.write(uploaded_file.getvalue())

                temp_path = Path(temp_file.name)

            # Load PDF.
            loader = PyPDFLoader(str(temp_path))

            documents = loader.load()

            if not documents:
                raise ValueError(f"No pages found in {uploaded_file.name}")

            # Add metadata.
            for document in documents:

                document.metadata["source"] = uploaded_file.name

                document.metadata["source_id"] = source_id

            # Split PDF into chunks.
            chunks = splitter.split_documents(documents)

            if not chunks:

                raise ValueError(
                    f"No readable text found in {uploaded_file.name}. "
                    "Scanned PDFs may require OCR."
                )

            # Store chunks in smaller batches.
            batch_size = 16

            for start in range(0, len(chunks), batch_size):

                batch = chunks[start : start + batch_size]

                vectorstore.add_documents(batch)

            processed.append(
                {
                    "id": source_id,
                    "name": uploaded_file.name,
                    "pages": len(documents),
                    "chunks": len(chunks),
                }
            )

        finally:

            if temp_path and temp_path.exists():

                temp_path.unlink(missing_ok=True)

    return processed


# ==================================================
# 8. DELETE DOCUMENT
# ==================================================


def delete_document(source_id):

    vectorstore = st.session_state.vectorstore

    if vectorstore is not None:

        vectorstore.delete(where={"source_id": source_id})

    st.session_state.documents.pop(source_id, None)

    st.session_state.selected_documents = [
        doc_id for doc_id in st.session_state.selected_documents if doc_id != source_id
    ]

    st.session_state.messages = []

    st.session_state.chat_document_key = None

    st.session_state.last_summary = None


# ==================================================
# 9. RETRIEVE DOCUMENTS
# ==================================================


def retrieve_documents(question, selected_ids):

    vectorstore = st.session_state.vectorstore

    if vectorstore is None:
        return []

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
            "filter": {"source_id": {"$in": selected_ids}},
        },
    )

    return retriever.invoke(question)


# ==================================================
# 10. CHAT HISTORY
# ==================================================


def get_chat_history():

    history = []

    # Exclude the latest question because it is passed separately.
    for message in st.session_state.messages[:-1][-8:]:

        role = message["role"].upper()

        history.append(f"{role}: {message['content']}")

    return "\n\n".join(history)


# ==================================================
# 11. GENERATE ANSWERS
# ==================================================


def generate_answer(question, selected_ids):

    docs = retrieve_documents(question, selected_ids)

    if not docs:

        return ("I could not find the answer in the uploaded documents.", [])

    context_parts = []

    sources = []

    for doc in docs:

        filename = doc.metadata.get("source", "Unknown document")

        page = doc.metadata.get("page")

        page_number = page + 1 if isinstance(page, int) else "Unknown"

        context_parts.append(f"""
            Document: {filename}
            Page: {page_number}

            Content:
            {doc.page_content}
            """)

        sources.append(
            {
                "filename": filename,
                "page": page_number,
                "content": doc.page_content,
            }
        )

    context = "\n\n".join(context_parts)

    final_prompt = qa_prompt.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": get_chat_history(),
        }
    )

    response = load_llm().invoke(final_prompt)

    return response.content, sources


# ==================================================
# 12. GET DOCUMENT CHUNKS
# ==================================================


def get_document_chunks(selected_ids):

    vectorstore = st.session_state.vectorstore

    if vectorstore is None:
        return []

    result = vectorstore.get(
        where={"source_id": {"$in": selected_ids}},
        include=["documents", "metadatas"],
    )

    chunks = []

    for content, metadata in zip(result["documents"], result["metadatas"]):

        chunks.append(
            {
                "content": content,
                "metadata": metadata,
            }
        )

    return chunks


# ==================================================
# 13. COMPLETE DOCUMENT SUMMARIZATION
# ==================================================


def summarize_documents(selected_ids):

    chunks = get_document_chunks(selected_ids)

    if not chunks:
        raise ValueError("No document content found.")

    model = load_llm()

    summaries = []

    progress = st.progress(0)

    status = st.empty()

    # MAP STEP
    # Summarize each document chunk.

    for index, chunk in enumerate(chunks):

        status.write(f"Summarizing section {index + 1} of {len(chunks)}...")

        result = model.invoke(map_prompt.invoke({"text": chunk["content"]}))

        summaries.append(result.content)

        progress.progress((index + 1) / len(chunks))

    # REDUCE STEP
    # Combine summaries in groups.

    while len(summaries) > 1:

        new_summaries = []

        for index in range(0, len(summaries), 8):

            batch = summaries[index : index + 8]

            combined = "\n\n".join(batch)

            result = model.invoke(reduce_prompt.invoke({"summaries": combined}))

            new_summaries.append(result.content)

        summaries = new_summaries

    progress.empty()

    status.empty()

    return summaries[0]


# ==================================================
# 14. MAIN HEADER
# ==================================================

st.title("📚 AskMyPDF")

st.markdown("""
    Upload research papers, ask questions and generate
    comprehensive summaries using AI.
    """)

st.caption("Your AI-powered document research assistant.")

st.divider()


# ==================================================
# 15. SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📂 Document Library")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Process Documents",
        use_container_width=True,
        disabled=not uploaded_files,
    ):

        try:

            with st.spinner("Processing your documents..."):

                processed = process_documents(uploaded_files)

                for document in processed:

                    st.session_state.documents[document["id"]] = document

                    if document["id"] not in st.session_state.selected_documents:

                        st.session_state.selected_documents.append(document["id"])

                st.session_state.messages = []

                st.session_state.chat_document_key = None

                st.session_state.last_summary = None

            st.success(f"{len(processed)} document(s) processed!")

            st.rerun()

        except Exception as error:

            st.error(f"Document processing failed: {error}")

    st.divider()

    st.subheader("Your Documents")

    if not st.session_state.documents:

        st.info("No documents uploaded yet.")

    else:

        for source_id, document in list(st.session_state.documents.items()):

            with st.expander(document["name"]):

                st.write(f"Pages: {document['pages']}")

                st.write(f"Chunks: {document['chunks']}")

                if st.button(
                    "Delete document",
                    key=f"delete_{source_id}",
                    use_container_width=True,
                ):

                    delete_document(source_id)

                    st.rerun()

    st.divider()

    if st.button("Clear conversation", use_container_width=True):

        st.session_state.messages = []

        st.session_state.last_summary = None

        st.rerun()


# ==================================================
# 16. DOCUMENT SELECTION
# ==================================================

if not st.session_state.documents:

    st.info("Upload a PDF from the sidebar to start chatting.")

    st.stop()


document_options = {
    document["name"] + " — " + source_id[:6]: source_id
    for source_id, document in st.session_state.documents.items()
}

selected_labels = st.multiselect(
    "Select documents for your conversation",
    options=list(document_options.keys()),
    default=[
        label
        for label, source_id in document_options.items()
        if source_id in st.session_state.selected_documents
    ],
)

selected_ids = [document_options[label] for label in selected_labels]

st.session_state.selected_documents = selected_ids


if not selected_ids:

    st.warning("Please select at least one document.")

    st.stop()


# Reset conversation when document selection changes.

selection_key = tuple(sorted(selected_ids))

if st.session_state.chat_document_key != selection_key:

    st.session_state.messages = []

    st.session_state.chat_document_key = selection_key

    st.session_state.last_summary = None


# ==================================================
# 17. DOCUMENT SUMMARIZATION UI
# ==================================================

st.subheader("📖 Document Summarization")

if st.button("Generate complete summary"):

    try:

        with st.spinner("Generating summary from all selected documents..."):

            summary = summarize_documents(selected_ids)

        st.session_state.last_summary = summary

    except Exception as error:

        st.error(f"Summarization failed: {error}")


if st.session_state.last_summary:

    with st.expander("📄 Complete document summary", expanded=True):

        st.markdown(st.session_state.last_summary)


st.divider()


# ==================================================
# 18. CHAT HISTORY DISPLAY
# ==================================================

st.subheader("💬 Chat with your documents")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("View sources"):

                for source in message["sources"]:

                    st.markdown(f"**{source['filename']} — " f"Page {source['page']}**")

                    st.write(source["content"])

                    st.divider()


# ==================================================
# 19. CHAT INPUT
# ==================================================

query = st.chat_input("Ask a question about your selected documents...")

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    with st.chat_message("assistant"):

        try:

            with st.spinner("Searching documents and generating answer..."):

                answer, sources = generate_answer(query, selected_ids)

            st.markdown(answer)

            if sources:

                with st.expander("View sources"):

                    for source in sources:

                        st.markdown(
                            f"**{source['filename']} — " f"Page {source['page']}**"
                        )

                        st.write(source["content"])

                        st.divider()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as error:

            error_message = f"Something went wrong: {error}"

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )
