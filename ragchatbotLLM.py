 # ragChatbot.py
import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# Page Config
st.title("🤖 C++ RAG Chatbot using Ollama")

# Load Data
loader = TextLoader(".../introduction.txt", encoding="utf-8")

@st.cache_resource
def load_vectorstore():
    # Load Document
    loader = TextLoader("CPP.txt", encoding="utf-8")
    documents = loader.load()

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents = text_splitter.split_documents(documents)

    #embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"

    )
    db = FAISS.from_documents(final_documents, embeddings)
    return db
    
db = load_vectorstore()

# load ollama LLM
llm= Ollama(model="gemma2:2b")
# chat interface
user_question = st.text_input("Ask a question about C++")
if user_question:
    with st.spinner('Thinking...'):
        docs=db.similarity_search(user_question)
        context = "\n".join([doc.page_content for doc in docs])

        prompt= f"""
        Answer the question using only the context below.
        

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """
        response = llm.invoke(prompt)
        st.subheader("Answer:")
        st.write(response)