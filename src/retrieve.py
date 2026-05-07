import numpy as np
import ollama

def query_embed(query):

    embedding=[]
    response=ollama.embeddings(model="nomic-embed-text",propmpt=query)
    embedding.append(np.array(response['embedding']).astype("float32"))

    return embedding


def retrieve(index,query_embedding,chunks,k=3):
    D, I = index.search(np.array([query_embedding]), k)
    
    return [chunks[i] for i in I[0]]