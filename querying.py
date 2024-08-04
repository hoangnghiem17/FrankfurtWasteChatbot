from langchain_community.vectorstores.chroma import Chroma

import os
import sys

# ------------------------------------ADD DOCUMENTS TO THE CHROMA STORE--------------------------
# 1) INVESTIGATE SEARCH QUERY IF DOCUMENTS CAN BE RETURNED BY CHROMADB
# 2) ADD DOCUMENTS to CHROMA STORE - https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
# 3) INVESTIGATE DB BASED ON CHROMA OBJECT
chroma_store = Chroma(
    persist_directory=os.path.join(os.getcwd(), "chroma"),
    collection_name="frankfurt_waste_chatbot_v1"
)


# Explore the documents - stored as dictionary 
documents = chroma_store.get()
print(type(documents))
print(type(documents[0]))
sys.exit()
for doc in documents:
    print("Document ID:", doc.metadata.get("id", "N/A"))
    print("Document Content:", doc.page_content)
    print("Document Metadata:", doc.metadata)
    print("-" * 40)
