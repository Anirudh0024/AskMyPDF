# 📚 PaperMind AI

### Your documents. Your questions. Intelligent answers.

PaperMind AI is an AI-powered document research assistant built with Python, LangChain, Streamlit, Hugging Face, ChromaDB, and Groq.

Upload multiple PDF documents, ask context-aware questions, explore source references, and generate comprehensive document summaries using Retrieval-Augmented Generation (RAG).

## ✨ Features

- 📄 Upload multiple PDF documents.
- 🔍 Ask questions about uploaded documents.
- 🤖 AI-powered answers using Groq LLM.
- 🧠 Semantic search using Hugging Face embeddings.
- 🗄️ Vector storage and retrieval using ChromaDB.
- 📖 Generate comprehensive document summaries.
- 📌 View source documents and page references.
- 💬 Interactive chat interface using Streamlit.
- 🧾 Maintain conversation history during a session.

## 🛠️ Tech Stack

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Programming language |
| LangChain    | RAG framework        |
| Streamlit    | User interface       |
| Hugging Face | Text embeddings      |
| ChromaDB     | Vector database      |
| Groq         | LLM inference        |
| PyPDF        | PDF document loading |

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/papermind-ai.git
cd papermind-ai
```

### 2. Install uv

Follow the official installation guide:

https://docs.astral.sh/uv/

### 3. Install dependencies

```bash
uv sync
```

### 4. Configure environment variables

Create a `.env` file in the project directory.

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the application

```bash
uv run streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🔄 How It Works

1. Upload PDF documents.
2. Extract text using PyPDFLoader.
3. Split documents into smaller chunks.
4. Generate embeddings using Hugging Face.
5. Store embeddings in ChromaDB.
6. Retrieve relevant chunks based on user queries.
7. Pass retrieved context to the Groq LLM.
8. Generate context-aware answers.

## 🧠 RAG Architecture

```text
PDF Documents
     |
     v
Document Loading
     |
     v
Text Splitting
     |
     v
Hugging Face Embeddings
     |
     v
ChromaDB Vector Store
     |
     v
User Question
     |
     v
Semantic Retrieval
     |
     v
Groq LLM
     |
     v
Context-Aware Answer
```

## 🔐 Environment Variables

Never commit your `.env` file or expose your API keys publicly.

## 🚀 Deployment

The application can be deployed using Render.

## 👨‍💻 Author

**Anirudh Pratap Singh**

Built as a hands-on project to explore Generative AI, LangChain, and Retrieval-Augmented Generation.
