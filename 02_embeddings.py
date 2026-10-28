
from sentence_transformers import SentenceTransformer

# Load a pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Our sample text chunks
chunks = [
    "Python is a programming language used for automation and data analysis.",
    "FAISS is a library for efficient similarity search over vectors.",
    "RAG retrieves relevant documents and supplies them to a language model.",
    "SentenceTransformers converts text into dense numerical embeddings."
]

# Convert text into numerical vectors
embeddings = model.encode(chunks)

# Inspect the results
print("Number of chunks:", len(chunks))
print("Embeddings shape:", embeddings.shape)

print("\nFirst chunk:")
print(chunks[0])

print("\nIts embedding (first 10 numbers):")
print(embeddings[0][:10])
