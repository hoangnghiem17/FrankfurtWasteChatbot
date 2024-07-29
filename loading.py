import os
from langchain_community.document_loaders import PyPDFLoader
import sys

def correct_ger_umlauts(text):
    replacements = {
        'Ã¤': 'ä',
        'Ã¶': 'ö',
        'Ã¼': 'ü',
        'Ã„': 'Ä',
        'Ã–': 'Ö',
        'Ãœ': 'Ü',
        'ÃŸ': 'ß'
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text

cur_dir = os.getcwd()
root_dir = os.path.join(cur_dir, "Dokumente", "Mülltrennung")

# Store PDFs in a list of dictionaries containing attributes for metadata
documents = [
    {"document_name": "FES_waskommtwohinein.pdf", "category": "mülltrennung_allgemein"},
    {"document_name": "FES_keinplastikindiebiotonne.pdf", "category": "mülltrennung_bio"},
    {"document_name": "MW_wertstofftonne.pdf", "category": "mülltrennung_wertstoff"}
]

#print(list(documents[0].items())[0][1]) # Access document name of first document

processed_docs = []

# Iterate over each document in list, read PDF and add category as new metadata attribute
for doc_info in documents:
    pdf = os.path.join(root_dir, doc_info["document_name"])
    
    # PyPDFLoader separates a document by page - access extracted text (page_content) or metadata (metadata)
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    
    valid_docs = []
    
    for doc in docs:
        clean_text = correct_ger_umlauts(doc.page_content)
        word_count = len(clean_text.split())
        
        if word_count > 10:
            doc.metadata["category"] = doc_info["category"]
            valid_docs.append(doc)
    
    processed_docs.extend(valid_docs)
        
"""
for doc in processed_docs:
    print(doc.metadata)

print(f"Total number of processed documents: {len(processed_docs)}")
print(processed_docs[4])
"""