"""
Generation step: build a prompt from retrieved chunks (with citations),
call an LLM, return the answer.
"""
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv("hugging.env")

def build_prompt(question: str, retrieved_chunks: list[dict]) -> list[dict]:
    """retrieved_chunks: list of {'text': ..., 'source': ...} dicts, already retrieved."""
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    system_msg = (
        "You are a financial analyst assistant. Answer the question using ONLY "
        "the context provided below. Cite the source filename in brackets for any claim, "
        "like [doc1.txt]. If the context doesn't contain the answer, say "
        "'I don't have enough information to answer that.'"
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

def call_llm(messages: list[dict], model: str = "meta-llama/Llama-3.1-8B-Instruct") -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in environment")
    client = InferenceClient(token=token)
    response = client.chat_completion(messages=messages, model=model, max_tokens=300)
    return response.choices[0].message.content
