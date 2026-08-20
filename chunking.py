def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    return chunks
