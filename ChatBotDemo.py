import streamlit as st
# import os
# from dotenv import load_dotenv
# import openai
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================
# Page Config
# =========================
st.set_page_config(page_title="C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("Ask any question related to C++ Introduction")

# Cache Document Loading
# ----------------------

@st.cache_resource
def load_vectorstore():
    # Load Document
    loader = TextLoader("cpp_introduction.txt", encoding="utf-8")
    documents = loader.load()

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    final_documents = text_splitter.split_documents(documents)

    # Create Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Create FAISS Vector Store
    db = FAISS.from_documents(final_documents, embeddings)
    return db

# User Input
# ----------------------------

query = st.text_input("Enter your question about the documents")

if query:
    docs = db.similarity_search(query, k=3)

    st.subheader("Retrieved Context:")

    for i, doc in enumerate(docs):
        st.markdown(f"*Result {i+1}:*")
        st.write(doc.page_content)
