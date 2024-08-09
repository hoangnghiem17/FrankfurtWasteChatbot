import os

import chromadb
from chromadb.config import Settings, DEFAULT_DATABASE, DEFAULT_TENANT

dev_directory = os.getcwd()
chroma_directory = os.path.join(dev_directory, "chroma")
document_directory = os.path.join(dev_directory, "Dokumente", "Mülltrennung")
embedding_model_name = "all-MiniLM-L6-v2"
gemma_token = "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"
api_url = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
chroma_client = chromadb.PersistentClient(
    path=chroma_directory,
    settings=Settings(allow_reset=True),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)