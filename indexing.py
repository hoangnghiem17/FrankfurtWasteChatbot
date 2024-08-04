from loading import preprocessed_docs
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
import os
import shutil
import sys
import logging

# Enable offline mode - Embedding model is not loaded again 
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Split documents into chunks and create embeddings
def chunk_documents(preprocessed_docs, chunk_size=500, chunk_overlap=100):
    """
    Splits preprocessed documents into smaller text chunks and creates document embeddings.

    This function takes a list of processed documents and splits each document's content into smaller chunks 
    using the `RecursiveCharacterTextSplitter`. Each chunk retains the original document's metadata and can be 
    used for further processing, such as creating embeddings for NLP tasks.

    Args:
        - preprocessed_docs (list of Document): A list of preprocessed documents from loading.py module.

        - chunk_size (int, optional): The maximum size of each text chunk. Default is 500 characters.

        - chunk_overlap (int, optional): The number of characters that should overlap between chunks. Default is 100 characters.

    Returns:
        list of Document: A list of LangChain `Document` objects, where each document contains a chunk of
        text and associated metadata from the original document.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []
    for doc in preprocessed_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            documents.append(Document(page_content=str(chunk), metadata=doc.metadata))
        
    return documents

# Embed and save chunks in Chroma vector store
def embed_and_store_chunks(chunked_documents, collection_name):
    """
    Embeds text chunks using the 'all-MiniLM-L6-v2' model and stores them in a Chroma vector store via LangChain.

    Args:
        - chunked_documents (list of Document): A list of document chunks as Langchain 'Document' objects.

        - collection_name (str): Name of the collection to store embeddings.

    Returns:
        Chroma: An instance of the Chroma vector store containing the stored embeddings.
    """
    
    # Initialize the embedding model using SentenceTransformer from LangChain
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    persist_directory = os.path.join(os.getcwd(), "chroma")
    
    # Delete the existing directory and its contents to avoid redundant data adding to Chroma vector store
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        logging.info(f"Deleted existing Chroma directory: {persist_directory}")
        
    # Use LangChain's Chroma.from_documents to embed and store
    chroma_store = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    return chroma_store

chunked_documents = chunk_documents(preprocessed_docs=preprocessed_docs)
chroma_store = embed_and_store_chunks(chunked_documents, collection_name="frankfurt_waste_chatbot_v1")

