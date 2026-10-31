
import streamlit as st
from rag_engine import load_documents, build_vectorstore, answer_question

from pathlib import Path


st.set_page_config(
    page_title="College Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 College Knowledge Assistant")
st.write(
    "Ask questions about college policies using "
    "your uploaded documents."
)

st.info(
    "Answers are generated from retrieved documents. "
    "Check the supporting sources before relying on a policy."
)

# Upload documents
uploaded_files = st.file_uploader(
    "Upload at least five PDF or TXT documents",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Build the searchable knowledge base
if st.button("Build Knowledge Base", type="primary"):
    if not uploaded_files or len(uploaded_files) < 5:
        st.error("Please upload at least five documents.")
    else:
        try:
            with st.spinner("Processing documents..."):
                documents = load_documents(uploaded_files)
                vectorstore, chunk_count = build_vectorstore(documents)

                st.session_state.vectorstore = vectorstore
                st.session_state.chat_history = []

            st.success(
                f"Knowledge base created! "
                f"{len(uploaded_files)} files processed into "
                f"{chunk_count} chunks."
            )

        except Exception as error:
            st.error(f"Could not process documents: {error}")

# Chat interface
if st.session_state.vectorstore is not None:
    st.subheader("Ask a question")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Supporting sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**File:** {source['source']}  \n"
                            f"**Page:** {source['page']}"
                        )
                        st.write(source["content"])

    question = st.chat_input(
        "Example: How many books can I borrow?"
    )

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents and generating answer..."):
                    result = answer_question(
                        st.session_state.vectorstore,
                        question,
                        k=6,
                    )

                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("Supporting sources"):
                        for source in result["sources"]:
                            st.markdown(
                                f"**File:** {source['source']}  \n"
                                f"**Page:** {source['page']}"
                            )
                            st.write(source["content"])

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )

            except Exception as error:
                st.error(f"Could not generate an answer: {error}")