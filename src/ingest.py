def chunking(path,chunk_size=700):
    with open(path, 'r', encoding='utf-8') as f:
        text=f.read()
    
    for chunk in text:
        chunks=[text[i:i+chunk_size] for i in range (0,len(text),chunk_size)]

    return chunks