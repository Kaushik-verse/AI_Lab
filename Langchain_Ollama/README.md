# PDF Chat with Ollama

A Streamlit app that allows you to **upload PDFs and ask questions** using **Ollama embeddings** and **LangChain**. The app uses AI to provide concise answers based on the content of your PDFs.

---

## Features

- Upload PDF files and store them locally
- Automatically split PDFs into smaller chunks for better AI understanding
- Generate embeddings using Ollama for semantic search
- Retrieve relevant information from PDFs to answer user queries
- AI-generated answers limited to 3 sentences for concise responses
- Interactive chat interface powered by Streamlit

---

## Technologies Used

- **Python 3.12+**
- **Streamlit** – Web app interface
- **LangChain** – Framework for LLMs and embeddings
- **Ollama** – Embeddings and LLM model (`deepseek-r1:1.5b`)
- **FAISS** – Vector store for semantic search
- **PyPDFLoader** – Load and read PDF documents

---

## Installation

1. **Clone this repository:**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git