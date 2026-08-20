def chunk_text(text: str ,chunk_size: int, overlap: int) -> list:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    chunking =[text[i:i + chunk_size] for i in range(0,len(text),chunk_size-overlap)]
    return chunking

# test
text = "ABCDEFGHIJ"  # 10 chars
result = chunk_text(text, chunk_size=4, overlap=1)
print("chunk_size=4, overlap=1:", result)

result2 = chunk_text(text, chunk_size=3, overlap=0)
print("chunk_size=3, overlap=0:", result2)

try:
    chunk_text(text, chunk_size=3, overlap=3)
except ValueError as e:
    print("Guard works:", e)

# edge case: empty text
print("empty text:", chunk_text("", 4, 1))
