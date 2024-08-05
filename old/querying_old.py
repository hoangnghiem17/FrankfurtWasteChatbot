from langchain_community.vectorstores.chroma import Chroma
import numpy as np
from collections import Counter
from chromadb.api.client import SharedSystemClient
from chromadb import Client

import os
import sys

# ------------------------------------ADD DOCUMENTS TO THE CHROMA STORE--------------------------
# 1) REBUILD USING CHROMADB DIRECTLY 
# 2) INVESTIGATE SEARCH QUERY IF DOCUMENTS CAN BE RETURNED BY CHROMADB
# 3) INVESTIGATE DB BASED ON CHROMA OBJECT

chroma_store = Chroma(
    persist_directory=os.path.join(os.getcwd(), "chroma"),
    collection_name="frankfurt_waste_chatbot_v1"
)

# Explore the documents - stored as dictionary with 7 keys (attributes)
documents = chroma_store.get()
for index, key in enumerate(documents.keys()):
    print(f"Index: {index}, Key: {key}")

# IDs
ids = documents['ids']
print("IDs Sample:", ids[:5])
print("Total IDs:", len(ids))
print("Unique IDs:", len(set(ids)))

# Embeddings - NO embeddings saved
"""
embeddings = documents['embeddings']
print("Embeddings Sample:", embeddings[:1])
print("Total Embeddings:", len(embeddings))
print("Embedding Dimension:", len(embeddings[0]))
"""

# Metadatas
metadatas = documents['metadatas']
print("Metadata Sample:", metadatas[:1])
metadata_keys = [list(meta.keys()) for meta in metadatas if meta]
unique_metadata_keys = set(sum(metadata_keys, []))
print("Unique Metadata Keys:", unique_metadata_keys)
metadata_counter = Counter(sum(metadata_keys, []))
print("Metadata Key Occurrences:", metadata_counter)

# Documents
docs = documents['documents']
print("Documents Sample:", docs[:1])
print("Total Documents:", len(docs))
doc_lengths = [len(doc) for doc in docs]
print("Average Document Length:", np.mean(doc_lengths))
print("Max Document Length:", np.max(doc_lengths))
print("Min Document Length:", np.min(doc_lengths))

# URIs - no URIs saved
"""
uris = documents['uris']
print("URIs Sample:", uris[:5])
print("Total URIs:", len(uris))
unique_domains = set(uri.split('/')[2] for uri in uris if uri)
print("Unique Domains:", unique_domains)
"""

# Data - no data saved
"""
data = documents['data']
print("Data Sample:", data[:1])
if isinstance(data[0], dict):
    unique_data_keys = set(sum([list(item.keys()) for item in data], []))
    print("Unique Data Keys:", unique_data_keys)
"""

# Included
included = documents['included']
print("Included Sample:", included[:5])
print("Total included:", len(included))

# Assuming 'documents' is your dictionary-like object from Chroma
first_document_info = {}

# Iterate through each key and get the first value
for key in documents.keys():
    # Handle list and dict types specifically
    if isinstance(documents[key], list):
        first_value = documents[key][0] if documents[key] else 'No data'
    elif isinstance(documents[key], dict):
        first_value = list(documents[key].items())[0] if documents[key] else 'No data'
    else:
        first_value = documents[key]  # For any other type, take the direct value

    first_document_info[key] = first_value

# Print the information of the first document
for key, value in first_document_info.items():
    print(f"Key: {key}\nValue: {value}\n")

