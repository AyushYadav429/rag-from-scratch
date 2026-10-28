
documents = [
    "Python is a programming language used for automation and data analysis.",
    "FAISS is a library for efficient similarity search over vectors.",
    "RAG retrieves relevant documents and supplies them to a language model.",
    "SentenceTransformers converts text into dense numerical embeddings."
]


def chunk_text(text, chunk_size=10):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


for document in documents:
    print(chunk_text(document))
