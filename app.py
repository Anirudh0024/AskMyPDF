from pathlib import Path
import tempfile
import uuid

import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


# ==================================================
# 2. LOAD MODELS
# ==================================================


@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():

    return init_chat_model("groq:openai/gpt-oss-120b")


# ==================================================
# 3. PROMPTS
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
        3. Use conversation history to understand follow-up questions.
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
# 4. SESSION STATE
# ==================================================

defaults = {
    "messages": [],
    "documents": {},
    "vectorstore": None,
    "db_path": None,
    "selected_documents": [],
    "chat_document_key": None,
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# 5. INITIALIZE VECTOR DATABASE
# ==================================================


def initialize_database():

    if st.session_state.vectorstore is not None:
        return st.session_state.vectorstore

    db_path = Path(tempfile.gettempdir()) / f"research_assistant_{uuid.uuid4().hex}"

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
# 6. PROCESS PDF DOCUMENTS
# ==================================================


def process_documents(uploaded_files):

    vectorstore = initialize_database()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    processed = []

    for uploaded_file in uploaded_files:

        source_id = uuid.uuid4().hex

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = Path(temp_file.name)

        try:

            # Load PDF
            loader = PyPDFLoader(str(temp_path))
            documents = loader.load()

            if not documents:
                raise ValueError(f"No pages found in {uploaded_file.name}")

            # Add metadata
            for document in documents:

                document.metadata["source"] = uploaded_file.name
                document.metadata["source_id"] = source_id

            # Split into chunks
            chunks = splitter.split_documents(documents)

            if not chunks:
                raise ValueError(f"No readable text found in {uploaded_file.name}")

            # Store in Chroma
            vectorstore.add_documents(chunks)

            processed.append(
                {
                    "id": source_id,
                    "name": uploaded_file.name,
                    "pages": len(documents),
                    "chunks": len(chunks),
                }
            )

        finally:

            temp_path.unlink(missing_ok=True)

    return processed


# ==================================================
# 7. DELETE DOCUMENT
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


# ==================================================
# 8. RETRIEVE DOCUMENTS
# ==================================================


def retrieve_documents(question, selected_ids):

    vectorstore = st.session_state.vectorstore

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
# 9. CHAT HISTORY
# ==================================================


def get_chat_history():

    history = []

    # Use recent conversation turns
    for message in st.session_state.messages[-8:]:

        role = message["role"].upper()

        history.append(f"{role}: {message['content']}")

    return "\n\n".join(history)


# ==================================================
# 10. ANSWER QUESTIONS
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
            {"filename": filename, "page": page_number, "content": doc.page_content}
        )

    context = "\n\n".join(context_parts)

    final_prompt = qa_prompt.invoke(
        {"context": context, "question": question, "chat_history": get_chat_history()}
    )

    response = load_llm().invoke(final_prompt)

    return response.content, sources


# ==================================================
# 11. GET ALL CHUNKS FOR SELECTED DOCUMENTS
# ==================================================


def get_document_chunks(selected_ids):

    vectorstore = st.session_state.vectorstore

    result = vectorstore.get(
        where={"source_id": {"$in": selected_ids}}, include=["documents", "metadatas"]
    )

    chunks = []

    for content, metadata in zip(result["documents"], result["metadatas"]):

        chunks.append({"content": content, "metadata": metadata})

    return chunks


# ==================================================
# 12. COMPLETE DOCUMENT SUMMARIZATION
# ==================================================


def summarize_documents(selected_ids):

    chunks = get_document_chunks(selected_ids)

    if not chunks:
        raise ValueError("No document content found.")

    model = load_llm()

    summaries = []

    progress = st.progress(0)
    status = st.empty()

    # MAP: Summarize every chunk
    for index, chunk in enumerate(chunks):

        status.write(f"Summarizing section {index + 1} of {len(chunks)}...")

        result = model.invoke(map_prompt.invoke({"text": chunk["content"]}))

        summaries.append(result.content)

        progress.progress((index + 1) / len(chunks))

    # REDUCE: Combine summaries in smaller groups
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
# 13. MAIN HEADER
# ==================================================

st.title("📚 AI Research Assistant")

st.markdown("""
    Upload research papers, ask questions and generate
    comprehensive summaries using AI.
    """)

st.divider()


# ==================================================
# 14. SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📂 Document Library")

    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True
    )

    if st.button(
        "Process Documents", use_container_width=True, disabled=not uploaded_files
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

            st.success(f"{len(processed)} document(s) processed!")

        except Exception as error:

            st.error(str(error))

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
        st.rerun()


# ==================================================
# 15. DOCUMENT SELECTION
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


# Reset chat when the document selection changes
selection_key = tuple(sorted(selected_ids))

if st.session_state.chat_document_key != selection_key:

    st.session_state.messages = []
    st.session_state.chat_document_key = selection_key


# ==================================================
# 16. SUMMARY FEATURE
# ==================================================

st.subheader("📖 Document Summarization")

if st.button("Generate complete summary", use_container_width=False):

    try:

        with st.spinner("Generating summary from all selected document chunks..."):

            summary = summarize_documents(selected_ids)

        st.session_state.last_summary = summary

    except Exception as error:

        st.error(f"Summarization failed: {error}")


if "last_summary" in st.session_state:

    with st.expander("📄 Complete document summary", expanded=True):

        st.markdown(st.session_state.last_summary)


st.divider()


# ==================================================
# 17. CHAT HISTORY DISPLAY
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
# 18. CHAT INPUT
# ==================================================

query = st.chat_input("Ask a question about your selected documents...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

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
                {"role": "assistant", "content": answer, "sources": sources}
            )

        except Exception as error:

            error_message = f"Something went wrong: {error}"

            st.error(error_message)

            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "sources": []}
            )
