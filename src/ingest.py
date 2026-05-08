def load_and_chunk(file_path, chunk_size=1500):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks