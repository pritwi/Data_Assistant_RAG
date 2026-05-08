import ollama

def generate_answer(context, query):
    prompt = f"""
You are a precise assistant.
Answer ONLY from the provided context.
If not found, say "Not found in context".

Context:
{context}

Question: {query}
Answer:
"""

    response = ollama.generate(
        model="gemma3:1b",
        prompt=prompt,
        options={"temperature": 0.1}
    )

    return response["response"]