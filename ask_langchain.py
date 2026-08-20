"""
LangChain-native version of the same RAG pipeline.
Same embedding model (all-mpnet-base-v2), same chunk settings, same LLM -
only the plumbing changes to LangChain's abstractions: Chroma vectorstore,
a retriever object, and an LCEL chain (retriever | prompt | llm | parser).
"""
import os
import glob
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv("hugging.env")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HF_TOKEN", "")

# 1. Load documents (LangChain's document loader instead of raw open())
docs = []
for filepath in sorted(glob.glob("data/*.txt")):
    docs.extend(TextLoader(filepath, encoding="utf-8").load())

# 2. Split (same recursive splitter, now producing LangChain Document objects with metadata)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# 3. Embed + store in Chroma (persists to disk, handles the vector index for us)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")

# 4. Retriever object - wraps similarity search behind a standard interface
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. LLM
llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=300,
    temperature=0.1,
))

# 6. Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial analyst assistant. Answer using ONLY the context below. "
               "Cite the source in brackets, like [doc1.txt]. If the answer isn't in the "
               "context, say 'I don't have enough information to answer that.'"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

def format_docs(retrieved_docs):
    return "\n\n".join(f"[Source: {d.metadata.get('source')}]\n{d.page_content}" for d in retrieved_docs)

# 7. LCEL chain: pipe retriever's output into format_docs, pass question through unchanged,
#    feed both into the prompt, then the llm, then parse to plain string
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    question = "What company is Meridian acquiring and what are the expected synergies?"
    print(f"\nQuestion: {question}\n")
    answer = chain.invoke(question)
    print(f"Answer:\n{answer}")
