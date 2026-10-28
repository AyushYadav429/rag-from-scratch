
from ollama import chat

# Our retrieved context
context = """
FAISS is a library for efficient similarity search over vectors.
It can search large collections of numerical vectors efficiently.
"""

# The user's question
question = "What is FAISS used for?"

# Construct the prompt
prompt = f"""
You are a helpful assistant answering questions using the
provided context.

Rules:
- Answer using only the information in the context.
- If the answer is not in the context, say you do not have
  enough information.
- Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""

# Send the prompt to Gemma through Ollama
response = chat(
    model="gemma:2b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# Display the generated answer
print("Question:", question)
print("\nGenerated answer:")
print(response.message.content)
