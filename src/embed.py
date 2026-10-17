import faiss
import numpy as np
import ollama

def create_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )
        embeddings.append(response["embedding"])
    
    return np.array(embeddings).astype("float32")


def build_faiss_index(embeddings):
    embeddings=np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim!=2:
        raise ValueError("Embeddings must be 2D array")

    if embeddings.shape[0]==0:
        raise ValueError("Embeddings cannot be empty")

    if embeddings.shape[1]==0:
        raise ValueError("Embedding Vectors cannot have zero dimension")

    index=faiss.indexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index