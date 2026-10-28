# app.py

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 RAG Document Assistant")

model_name = st.sidebar.text_input("Ollama Model", "gemma2:2b")
top_k = st.sidebar.slider("Retrieved chunks", 1, 6, 3)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file:
    if st.button("Process Document"):
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(
                    temp_path, encoding="utf-8"
                )

            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(documents)

            vector_db = FAISS.from_documents(
                chunks, get_embeddings()
            )

            st.session_state["vector_db"] = vector_db
            st.success("Document processed successfully!")

        except Exception as e:
            st.error(f"Error processing document: {e}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if "vector_db" in st.session_state:
    question = st.text_input("Ask a question about your document")

    if st.button("Get Answer", disabled=not question.strip()):
        try:
            with st.spinner("Generating answer..."):
                docs = st.session_state["vector_db"].similarity_search(
                    question, k=top_k
                )

                context = "\n\n".join(
                    doc.page_content for doc in docs
                )

                prompt = ChatPromptTemplate.from_template(
                    """You are a helpful assistant.
Answer using only the following context.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}

Answer:"""
                )

                llm = OllamaLLM(
                    model=model_name,
                    temperature=0
                )

                chain = (
                    prompt
                    | llm
                    | StrOutputParser()
                )

                answer = chain.invoke({
                    "context": context,
                    "question": question
                })

            st.subheader("Answer")
            st.write(answer)

            with st.expander("View retrieved documents"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(doc.page_content)

        except Exception as e:
            st.error(f"Error generating answer: {e}")