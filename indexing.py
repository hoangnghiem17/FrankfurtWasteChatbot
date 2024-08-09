import os
import sys
import logging
import shutil
from typing import List

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from chromadb import Client

from loading import preprocess_docs
from config import dev_directory, chroma_directory, embedding_model_name, chroma_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

embedding_function = SentenceTransformer(embedding_model_name)

# Chroma settings - ensure directory exists
os.makedirs(chroma_directory, exist_ok=True)

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
    
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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
    embeddings = embedding_function.encode(texts, convert_to_tensor=False)
    embeddings = [embedding.tolist() for embedding in embeddings]  # Convert ndarray to list for easer adding to chroma vector store
    
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
        
        collection = chroma_client.create_collection(name=collection_name) #embedding_function
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
    
    # Define document paths
    documents = [
        {"document_name": "FES_waskommtwohinein.pdf", "category": "mülltrennung_allgemein"},
        {"document_name": "FES_keinplastikindiebiotonne.pdf", "category": "mülltrennung_bio"},
        {"document_name": "MW_wertstofftonne.pdf", "category": "mülltrennung_wertstoff"}
    ]

    try:
        # Preprocess raw documents
        preprocessed_docs = preprocess_docs(documents=documents, root_dir=dev_directory)
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
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        sys.exit(1)
