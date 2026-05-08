import numpy as np
import ollama

def get_query_embedding(query):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )
    return np.array(response["embedding"]).astype("float32")


def retrieve(index, query_embedding, chunks, k=6):
    D, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]