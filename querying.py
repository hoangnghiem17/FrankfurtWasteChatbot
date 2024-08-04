from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

import os
import sys

# ------------------------------------ADD DOCUMENTS TO THE CHROMA STORE--------------------------
# 1) ADD DOCUMENTS to CHROMA STORE - https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
# 2) INVESTIGATE DB BASED ON CHROMA OBJECT
chroma_store = Chroma(
    persist_directory=os.path.join(os.getcwd(), "chroma"),
    collection_name="frankfurt_waste_chatbot_v1"
)


# Explore the documents
documents = chroma_store.get()
for doc in documents:
    print("Document ID:", doc.metadata.get("id", "N/A"))
    print("Document Content:", doc.page_content)
    print("Document Metadata:", doc.metadata)
    print("-" * 40)
