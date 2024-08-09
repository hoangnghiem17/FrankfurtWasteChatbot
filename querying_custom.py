from config import chroma_directory, chroma_client

import chromadb


def load_chroma_collection(name):
    """
    Loads an existing Chroma collection from the specified path with the given name.

    Parameters:
    - path (str): The path where the Chroma database is stored.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - chromadb.Collection: The loaded Chroma Collection.
    """
    db = chroma_client.get_collection(name=name) #embedding_function=GeminiEmbeddingFunction())

    return db

db = load_chroma_collection(name="frankfurt_waste_chatbot_v1")
print(db)

def get_relevant_documents(query, db, n_results):
  passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
  return passage

relevant_text = get_relevant_documents(query="Biomüll",db=db,n_results=3)
# Print each relevant text in a separate line with its index
print(type(relevant_text))
for index, text in enumerate(relevant_text):
    print(f"{index + 1}. {text}")