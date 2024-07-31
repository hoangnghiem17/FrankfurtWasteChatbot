from loading import processed_docs
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
import os
import sys

# Enable offline mode - Embedding model is not loaded again 
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Split documents into chunks and create embeddings
def chunk_documents(processed_docs, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []
    for doc in processed_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            documents.append(Document(page_content=str(chunk), metadata=doc.metadata))
        
    return documents

chunked_documents = chunk_documents(processed_docs=processed_docs)
for doc in chunked_documents:
    print(doc.metadata)
sys.exit()


# # Print embedded chunks
# print(chunks_embeddings.shape)
# for i, chunk in enumerate(chunks_embeddings):
#     print(f"Chunk {i+1}:\n{chunk}\n")

chunks = chunk_documents(processed_docs=processed_docs)

documents = [Document(page_content=chunk) for chunk in chunks]

for doc in documents:
    print(doc.metadata)
print(f"Total number of processed documents: {len(documents)}")
print(documents[4])
sys.exit()
"""
# Print chunks
print(processed_docs[4].metadata)
print(len(chunks))
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
"""

# Store chunks and embeddings in Chroma vector store
embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_store = Chroma.from_documents(chunks, embeddings_model)

all_docs = chroma_store.get(ids=0)
print(all_docs)