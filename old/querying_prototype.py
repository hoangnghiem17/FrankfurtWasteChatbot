from loading import preprocess_docs
from indexing import chunk_documents, embed_documents 

from chromadb import Client
from chromadb.utils import embedding_functions
import chromadb
from chromadb.config import Settings, DEFAULT_DATABASE, DEFAULT_TENANT

import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

# Define documents
cur_dir = os.getcwd()
root_dir = os.path.join(cur_dir, "Dokumente", "Mülltrennung")

documents = [
    {"document_name": "FES_waskommtwohinein.pdf", "category": "mülltrennung_allgemein"},
    {"document_name": "FES_keinplastikindiebiotonne.pdf", "category": "mülltrennung_bio"},
    {"document_name": "MW_wertstofftonne.pdf", "category": "mülltrennung_wertstoff"}
]
    
# Preprocess raw documents
preprocessed_docs = preprocess_docs(documents=documents, root_dir=root_dir)
logging.info("Preprocessing completed.")

# Split preprocessed documents into chunks
chunked_documents = chunk_documents(preprocessed_docs=preprocessed_docs)

# Embed chunks
embeddings = embed_documents(chunked_documents)

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma")
COLLECTION_NAME = "frankfurt_waste_chatbot_v1"

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(
    path=PERSIST_DIRECTORY,
    settings=Settings(allow_reset=True),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)

# Step 2: Create a collection
chroma_client.reset()
collection_name = "test_db"
collection = chroma_client.create_collection(
    name=collection_name
    #metadata={"hnsw:space": "cosine"} # change distance function - l2 is the default

)

# Insert documents and embeddings into collection
for idx, (doc, embed) in enumerate(zip(chunked_documents, embeddings)):
    collection.add(
        embeddings=[embed],          # Add embedding as list of floats
        documents=[doc.page_content],  # Add document content
        metadatas=[doc.metadata],  # Add metadata (optional)
        ids=[str(idx)]                 # Use the index as a unique ID
    )

# Investigate insertion by querying the collection - embeddings is expected to be None
all_documents = collection.get()

# Print the total number of documents and their content
print(f"Total Documents: {len(all_documents['documents'])}")

# Print each document content and metadata
for idx, (document, metadata) in enumerate(zip(all_documents["documents"], all_documents["metadatas"])):
    print(f"Document {idx} Metadata: {metadata}")
    print(f"Document {idx} Content: {document}\n")

# Query the collection to return n most similar results
results = collection.query(
    query_texts=["Das ist eine Query über Biomüll."], # Chroma embeds this 
    n_results=3,
    include=['documents', 'metadatas', 'distances'] #, 'embeddings']
    #where={"metadata_field": "is_equal_to_this"}, # Filter by metadata 
    #where_document={"$contains":"search_string"} # Filter by document
)
print(results)
