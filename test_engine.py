
from pathlib import Path
from rag_engine import load_documents, build_vectorstore, answer_question


class SampleFile:
    def __init__(self, path):
        self.name = path.name
        self.path = path

    def getvalue(self):
        return self.path.read_bytes()


# Load the sample text files from your current folder
folder = Path(__file__).parent

files = [
    SampleFile(path)
    for path in folder.glob("*.txt")
    if path.name != "expected_test_questions.txt"
    and "conflict" not in path.name.lower()
]

print("Files found:", len(files))

documents = load_documents(files)
print("Documents loaded:", len(documents))

vectorstore, chunk_count = build_vectorstore(documents)

print("Knowledge base created successfully!")
print("Total text chunks:", chunk_count)

question = input("\nAsk a question about college documents: ")

result = answer_question(vectorstore, question)

print("\nANSWER:")
print(result["answer"])

print("\nSUPPORTING SOURCES:")
for source in result["sources"]:
    print(f"\nFile: {source['source']}")
    print(f"Page: {source['page']}")
    print(source["content"][:300])
    print("-" * 40)