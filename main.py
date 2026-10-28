
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import chat


# ==========================================
# 1. KNOWLEDGE BASE
# ==========================================

documents = [
    "Python is a programming language used for automation and data analysis.",
    "FAISS is a library for efficient similarity search over vectors.",
    "RAG retrieves relevant documents and supplies them to a language model.",
    "SentenceTransformers converts text into dense numerical embeddings.",
    "An embedding represents text as a numerical vector.",
    "A vector database or index helps retrieve relevant information.",
    "Chunking divides long documents into smaller pieces of text.",
    "BM25 is a lexical retrieval algorithm that ranks documents using term statistics."
]


# ==========================================
# 2. CHUNKING
# ==========================================

def chunk_text(text, chunk_size=40):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


chunks = []

for document in documents:
    chunks.extend(chunk_text(document))

print(f"Total chunks created: {len(chunks)}")


# ==========================================
# 3. EMBEDDINGS
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedding_model.encode(chunks)

embeddings = np.asarray(embeddings, dtype="float32")

print("Embedding matrix shape:", embeddings.shape)


# ==========================================
# 4. FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# ==========================================
# 5. USER QUESTION
# ==========================================

question = input("\nAsk a question: ")


# ==========================================
# 6. RETRIEVAL
# ==========================================

query_embedding = embedding_model.encode([question])

query_embedding = np.asarray(
    query_embedding,
    dtype="float32"
)

k = min(3, len(chunks))

distances, indices = index.search(query_embedding, k)

retrieved_chunks = []

print("\nRetrieved context:")

for rank, idx in enumerate(indices[0], start=1):
    chunk = chunks[idx]
    retrieved_chunks.append(chunk)

    print(f"\n{rank}. {chunk}")
    print(f"   L2 distance: {distances[0][rank - 1]:.4f}")


# ==========================================
# 7. PROMPT CONSTRUCTION
# ==========================================

context = "\n".join(retrieved_chunks)

prompt = f"""
You are a helpful assistant answering questions using retrieved context.

Instructions:
- Use the provided context to answer the question.
- Do not invent facts that are not supported by the context.
- If the context does not contain enough information, say so.
- Answer clearly and concisely.

Retrieved context:
{context}

User question:
{question}

Answer:
"""


# ==========================================
# 8. GENERATION
# ==========================================

print("\nGenerating answer with Gemma 2B...")

response = chat(
    model="gemma:2b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ==========================================
# 9. FINAL ANSWER
# ==========================================

print("\n" + "=" * 50)
print("RAG ANSWER")
print("=" * 50)

print(response.message.content)
