import os
import sys
import logging
import shutil
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb
from chromadb import Client
from chromadb.config import Settings, DEFAULT_DATABASE, DEFAULT_TENANT
from loading import preprocess_docs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

# Configure SentenceTransformerModel
model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma settings
persist_directory = os.path.join(os.getcwd(), "chroma")
os.makedirs(persist_directory, exist_ok=True)  # Ensure directory exists

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(allow_reset=True),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)

def chunk_documents(preprocessed_docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """
    Splits preprocessed documents into smaller text chunks.

    Args:
        - preprocessed_docs (List[Document]): A list of preprocessed documents.
        - chunk_size (int, optional): Maximum size of each text chunk. Default is 500 characters.
        - chunk_overlap (int, optional): Number of characters to overlap between chunks. Default is 100 characters.

    Returns:
        - List[Document]: A list of documents containing text chunks and associated metadata.
    """
    
    logging.info("Starting document chunking process.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []

    for doc in preprocessed_docs:
        try:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                documents.append(Document(page_content=str(chunk), metadata=doc.metadata))
        except Exception as e:
            logging.error(f"Error while splitting document {doc.metadata.get('document_name', 'unknown')}: {e}")

    logging.info(f"Document chunking completed. Total chunks created: {len(documents)}")
    
    return documents

def embed_documents(documents: List[Document]) -> List[List[float]]:
    """
    Embeds text chunks using the specified embedding model.

    Args:
        - documents (List[Document]): A list of document chunks.

    Returns:
        - List[List[float]]: A list of embeddings for each document chunk.
    """
    
    logging.info("Starting embedding process.")
    texts = [doc.page_content for doc in documents]
    embeddings = model.encode(texts, convert_to_tensor=False)
    embeddings = [embedding.tolist() for embedding in embeddings]  # Convert ndarray to list
    logging.info("Embedding process completed.")
    logging.debug(f"Generated {len(embeddings)} embeddings.")

    return embeddings

def store_embeddings_in_chroma(documents: List[Document], embeddings: List[List[float]], collection_name: str):
    """
    Stores embeddings in a Chroma vector store.

    Args:
        - documents (List[Document]): A list of document chunks from Documents Class.
        - embeddings (List[List[float]]): Embeddings for each document chunk.
        - collection_name (str): Name of the Chroma collection.
    """
    
    try:
        chroma_client.reset()
        logging.debug(f"Chroma database resetted.")

        collection = chroma_client.create_collection(name=collection_name)
        logging.info("Collection created in Chroma.")

        for idx, doc in enumerate(documents):
            metadata = doc.metadata or {}
            collection.add(
                ids=[f"doc_{idx}"],
                embeddings=[embeddings[idx]],
                metadatas=[metadata],
                documents=[doc.page_content]
            )
        logging.info("Embeddings stored in Chroma.")

    except Exception as e:
        logging.error(f"Error storing embeddings in Chroma: {e}")
    
    return collection

if __name__ == "__main__":
    
    # Define documents
    cur_dir = os.getcwd()
    root_dir = os.path.join(cur_dir, "Dokumente", "Mülltrennung")

    documents = [
        {"document_name": "FES_waskommtwohinein.pdf", "category": "mülltrennung_allgemein"},
        {"document_name": "FES_keinplastikindiebiotonne.pdf", "category": "mülltrennung_bio"},
        {"document_name": "MW_wertstofftonne.pdf", "category": "mülltrennung_wertstoff"}
    ]

    try:
        # Preprocess raw documents
        preprocessed_docs = preprocess_docs(documents=documents, root_dir=root_dir)
        logging.info("Preprocessing completed.")

        # Split preprocessed documents into chunks
        chunked_documents = chunk_documents(preprocessed_docs=preprocessed_docs)

        # Embed chunks
        embeddings = embed_documents(chunked_documents)

        
        # Check if documents and embeddings match
        if len(chunked_documents) != len(embeddings):
            logging.error("Mismatch between number of documents and embeddings.")
            sys.exit(1)

        # Store embeddings in Chroma
        collection_name = "frankfurt_waste_chatbot_v1"
        collection = store_embeddings_in_chroma(chunked_documents, embeddings, collection_name)
        
        # Validation
        print(f"Total chunks created: {len(chunked_documents)}")
        print(f"Total embeddings generated: {len(embeddings)}")
        
        # Retrieve data from the collection for sanity checks
        retrieved_data = collection.get()
        retrieved_embeddings = retrieved_data['embeddings']
        retrieved_documents = retrieved_data['documents']

        # Check retrieved data lengths
        print(f"Total documents in collection: {len(retrieved_documents)}")
        #print(f"Total embeddings in collection: {len(retrieved_embeddings)}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        sys.exit(1)
