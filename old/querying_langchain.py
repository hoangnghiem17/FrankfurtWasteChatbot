import os
import sys
import logging
import shutil
from typing import List

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

from loading import preprocess_docs
from indexing import chunk_documents, embed_documents, store_embeddings_in_chroma
from gemma_api import query_llm
from config import dev_directory, chroma_client, embedding_model_name

# Define document paths
documents = [
    {"document_name": "FES_waskommtwohinein.pdf", "category": "mülltrennung_allgemein"},
    {"document_name": "FES_keinplastikindiebiotonne.pdf", "category": "mülltrennung_bio"},
    {"document_name": "MW_wertstofftonne.pdf", "category": "mülltrennung_wertstoff"}
]

try:
    # Preprocess raw documents
    preprocessed_docs = preprocess_docs(documents=documents, root_dir=dev_directory)

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
    
# Intialize Chroma vector store as retriever
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

chroma_retriever = Chroma(
    client=chroma_client,
    collection_name=collection_name,
    embedding_function=embedding_function
).as_retriever() # add parameters

prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
    You are an expert assistant specializing in waste management. Use the following information to answer the user's question in German.

    Context: {context}

    Question: {input}

    Please provide a detailed and helpful response based on the above context.
    """
)

# Create document combination chain to pass retrieved documents and formatted prompt to LLM
combine_docs_chain = create_stuff_documents_chain(
    llm=query_llm,
    prompt=prompt    
)

retrieval_chain = create_retrieval_chain(
    retriever=chroma_retriever, # To fetch relevant documents
    combine_docs_chain=combine_docs_chain # Chain combining documents and queries
)

# Execute the chain with a sample query
query = "What should I do with plastic waste?"
result = retrieval_chain.invoke({"input": query})

print(result)
sys.exit()
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