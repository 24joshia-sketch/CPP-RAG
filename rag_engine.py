
from io import BytesIO

import ollama
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


def load_documents(uploaded_files):
    documents = []

    for file in uploaded_files:
        filename = file.name
        content = file.getvalue()

        if filename.lower().endswith(".txt"):
            text = content.decode("utf-8", errors="replace")

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": 1,
                        },
                    )
                )

        elif filename.lower().endswith(".pdf"):
            reader = PdfReader(BytesIO(content))

            for page_number, page in enumerate(
                reader.pages, start=1
            ):
                text = page.extract_text() or ""

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_number,
                            },
                        )
                    )

    return documents


def build_vectorstore(documents):
    if not documents:
        raise ValueError(
            "No readable text found in the uploaded files."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were generated.")

    vectorstore = FAISS.from_documents(
        chunks, embeddings
    )

    return vectorstore, len(chunks)


def answer_question(vectorstore, question, k=4):
    results = vectorstore.similarity_search_with_score(
        question, k=k
    )

    if not results:
        return {
            "answer": (
                "I could not find sufficient information "
                "in the uploaded documents."
            ),
            "sources": [],
        }

    sources = []
    context_parts = []

    for index, (doc, distance) in enumerate(
        results, start=1
    ):
        source = doc.metadata.get(
            "source", "Unknown document"
        )
        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"[Source {index}: {source}, page {page}]\n"
            f"{doc.page_content}"
        )

        sources.append({
            "source": source,
            "page": page,
            "distance": round(float(distance), 4),
            "content": doc.page_content,
        })

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a college knowledge assistant.

Answer using ONLY the supplied document context.
Do not use outside knowledge or invent facts.
If the context does not contain enough information,
say that the documents do not provide the answer.
If documents disagree, explain the disagreement
and identify both sources.
Cite evidence using the source labels provided below.

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={"temperature": 0},
    )

    answer = response["message"]["content"].strip()

    return {
        "answer": answer,
        "sources": sources,
    }