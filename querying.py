import os
import logging
import chromadb
from chromadb.config import Settings, DEFAULT_DATABASE, DEFAULT_TENANT

# Configure logging with filename and line number
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

# Chroma settings
persist_directory = os.path.join(os.getcwd(), "chroma")

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)

def load_collection(collection_name: str):
    """
    Loads a collection from the Chroma database.

    Args:
        - collection_name (str): Name of the Chroma collection to load.

    Returns:
        - The collection instance from Chroma.
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
        logging.info(f"Collection '{collection_name}' loaded successfully.")
        return collection
    except Exception as e:
        logging.error(f"Error loading collection '{collection_name}': {e}")
        return None
    
def investigate_data(collection):
    """
    Investigates and prints details of the data in the collection.

    Args:
        - collection: The Chroma collection instance to investigate.
    """
    try:
        if collection is None:
            logging.error("No collection provided for investigation.")
            return

        # Retrieve all documents from the collection
        stored_data = collection.get()

        if not stored_data or 'documents' not in stored_data or not stored_data['documents']:
            logging.error("No data found in the collection.")
            return

        # Print details of the stored data
        print("\nInvestigation of Stored Data:")
        for i, (stored_doc, stored_metadata, stored_embedding) in enumerate(
            zip(stored_data['documents'], stored_data['metadatas'], stored_data['embeddings'])
        ):
            print(f"Document {i + 1}: {stored_doc[:75]}... (Metadata: {stored_metadata})")
            print(f"Embedding {i + 1}: {stored_embedding[:5]}... (Length: {len(stored_embedding)})\n")

        logging.info(f"Successfully investigated {len(stored_data['documents'])} documents in the collection.")
    except Exception as e:
        logging.error(f"Error during investigation: {e}")
        
if __name__ == "__main__":
    # Specify the collection name to investigate
    collection_name = "frankfurt_waste_chatbot_v1"

    # Load the collection
    collection = load_collection(collection_name)
    
    # Investigate the data within the collection
    investigate_data(collection)