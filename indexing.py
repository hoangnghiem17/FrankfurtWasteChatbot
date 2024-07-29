from loading import processed_docs
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
import sys

# Enable offline mode - Embedding model is not loaded again 
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Split documents into chunks and create embeddings
def chunk_documents(processed_docs, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in processed_docs:
        doc_chunks = text_splitter.split_text(doc.page_content)
        chunks.extend(doc_chunks)
        
    return chunks

chunks = chunk_documents(processed_docs=processed_docs)

"""
# Print chunks
print(processed_docs[4].metadata)
print(len(chunks))
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
"""

# Embed chunks
def embed_chunks(chunks): 
    embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embeddings_model.encode(chunks)
    return embeddings

chunks_embeddings = embed_chunks(chunks=chunks)
"""
# Print embedded chunks
print(chunks_embeddings.shape)
for i, chunk in enumerate(chunks_embeddings):
    print(f"Chunk {i+1}:\n{chunk}\n")
"""

# Store chunks embeddings in vector store