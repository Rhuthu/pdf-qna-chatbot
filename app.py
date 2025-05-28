import streamlit as st
from utils.pdf_reader import extract_text_from_pdf
from utils.chunker import chunk_text
from utils.embedding_generator import generate_embeddings, model
from utils.retriever import build_faiss_index, search_index
from utils.llm_answer import generate_llm_response
from utils.summarizer import better_summarize_pdf  



import tempfile
import os



# Set page config
st.set_page_config(page_title="📄 PDF Q&A Chatbot", layout="wide")
st.title("📄 PDF Q&A Chatbot")
st.caption("Ask questions from any PDF using Retrieval-Augmented Generation (RAG)")



# Session state tracker for uploaded file
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_uploaded:
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.pop("query", None)
        st.session_state.pop("summary", None)
        st.session_state.pop("chunks", None)
        st.session_state.pop("index", None)
        st.session_state.pop("embeddings", None)
    st.sidebar.info(f"📄 File: {uploaded_file.name}\n📦 Size: {uploaded_file.size // 1024} KB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(tmp_path)

    if not extracted_text.strip():
        st.error("No text could be extracted from this PDF.")
        st.stop()

    @st.cache_resource
    def summarize_text(text):
        return better_summarize_pdf(text[:5000])  # limit input to avoid slowness

    with st.spinner("Generating abstract summary..."):
        summary = summarize_text(extracted_text)
        st.subheader("📄 Abstract (Auto Summary):")
        st.write(summary)
    
    with st.spinner("Chunking text..."):
        chunks = chunk_text(extracted_text)

    @st.cache_resource
    def get_index(chunks):
        embeddings = generate_embeddings(chunks)
        return build_faiss_index(embeddings), embeddings
    

    with st.spinner("Generating embeddings and building index..."):
        index, embeddings = get_index(chunks)

    
    # Query input
    query = st.text_input("Ask a question about the PDF content:", value=st.session_state.get("query", ""))
    st.session_state["query"] = query

    if query:
        with st.spinner("Retrieving relevant chunks..."):
            query_embedding = model.encode(query)
            top_indices, scores = search_index(index, query_embedding, top_k=3)
            top_chunks = [chunks[idx] for idx in top_indices]
            context = "\n\n".join(top_chunks)

        with st.spinner("Generating answer with LLM..."):
            try:
                answer = generate_llm_response(query, context)
                st.subheader("💬 Answer:")
                st.write(answer)
                
              
            except Exception as e:
                st.error("Failed to get response from LLM Foundry.")
                st.exception(e)

        with st.expander("Retrieved Chunks + Scores"):
            for i, (chunk, score) in enumerate(zip(top_chunks, scores)):
                st.markdown(f"**Chunk {i+1}** (Similarity Score: {score:.2f})")
                st.write(chunk)
        
        