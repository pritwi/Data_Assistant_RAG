import faiss
import numpy as np
import ollama


def create_embedding(chunk):
    
    embeddings=[]    
    response = ollama.embeddings(model="nomic-embed-text",prompt=chunk)
    embeddings.append(response["embedding"])

    return np.array(embeddings).astype("float32")


def indexing(embeddings):

    dim=embeddings.shape[1]
    index=faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index