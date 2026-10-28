
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Our knowledge base
chunks = [
    "Python is a programming language used for automation and data analysis.",
    "FAISS is a library for efficient similarity search over vectors.",
    "RAG retrieves relevant documents and supplies them to a language model.",
    "SentenceTransformers converts text into dense numerical embeddings."
]

# 3. Convert documents into vectors
embeddings = model.encode(chunks)

# FAISS expects a NumPy array of float32 values
embeddings = np.asarray(embeddings, dtype="float32")

# 4. Create a FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# 5. Store the vectors in the index
index.add(embeddings)

print("Number of vectors stored:", index.ntotal)
print("Vector dimensions:", dimension)

# 6. Ask a question
query = "What is FAISS used for?"

# 7. Convert the question into a vector
query_embedding = model.encode([query])
query_embedding = np.asarray(query_embedding, dtype="float32")

# 8. Search for the 2 nearest vectors
distances, indices = index.search(query_embedding, k=2)

# 9. Display the results
print("\nQuestion:", query)
print("\nRetrieved documents:")

for rank, (distance, idx) in enumerate(
    zip(distances[0], indices[0]), start=1
):
    print(f"\nRank: {rank}")
    print(f"Distance: {distance:.4f}")
    print(f"Document: {chunks[idx]}")
