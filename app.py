import streamlit as st
from src.ingest import load_and_chunk
from src.embed import create_embeddings, build_faiss_index
from src.retrieve import get_query_embedding, retrieve
from src.generate import generate_answer

st.title("📄 AI Document Assistant")

uploaded_file = st.file_uploader("Upload a text file")

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    
    chunks = [text[i:i+700] for i in range(0, len(text), 1500)]
    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    query = st.text_input("Ask a question")

    if query:
        query_embedding = get_query_embedding(query)
        retrieved_chunks = retrieve(index, query_embedding, chunks)

        context = "\n---\n".join(retrieved_chunks)
        answer = generate_answer(context, query)

        st.write("### Answer:")
        st.write(answer)