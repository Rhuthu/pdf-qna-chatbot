# PDF Q&A Chatbot using RAG (Retrieval-Augmented Generation)

This is a **Streamlit-based PDF Q&A chatbot** that allows users to upload any PDF file, automatically generates an abstract summary, and enables them to ask questions about the document content using an LLM powered by Retrieval-Augmented Generation (RAG).

---

## Features

- Upload any PDF document.
- Automatically extract and chunk the text.
- Generate a high-level **abstract summary** of the document.
- Ask questions about the PDF using **LLM + vector similarity search (FAISS)**.
- Retrieve the top relevant chunks used to generate each answer.
- Friendly web-based UI using **Streamlit**.
- Secure token management using `.env` file (not exposed in Git).

---

## Tech Stack

- **Python**
- **Streamlit** – Web UI
- **SentenceTransformers** – Embedding model
- **FAISS** – Similarity search
- **HuggingFace / LLM Foundry API** – LLM for Q&A
- **PyMuPDF / pdfplumber** – PDF text extraction


1. Clone the repo
```bash

git clone https://github.com/Rhuthu/pdf-qna-chatbot.git

cd pdf-qna-chatbot

python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

pip install -r requirements.txt
```
Add your .env with the LLM_TOKEN