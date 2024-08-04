import os
from langchain_community.document_loaders import PyPDFLoader
import sys

def correct_ger_umlauts(text):
    """
    Corrects incorrectly encoded German umlauts and the Eszett in a given text.

    Args:
        text (str): The input string that may contain incorrectly encoded umlauts and the Eszett.

    Returns:
        str: A string with the incorrect characters replaced by the correct German umlauts and Eszett.
    """
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


def preprocess_docs(documents, root_dir):
    """
    Processes a list of PDF documents by:
        - splitting into pages
        - correcting text encoding errors
        - adds metadata attributes (document_name, category)
        - filters by documents with > 10 words

    Args:
        - documents (list of dict): A list of dictionaries where each dictionary contains information about a document, specifically:
            - 'document_name': The name of the document file (str).
            - 'category': The category to be assigned to each document (str).

        - root_dir (str): The root directory where the PDF documents are stored.

    Returns:
        list: A list of processed documents with added metadata and corrected text.
    """
    
    preprocessed_docs = []

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
                doc.metadata["document_name"] = doc_info["document_name"]
                valid_docs.append(doc)

        preprocessed_docs.extend(valid_docs)

    return preprocessed_docs

cur_dir = os.getcwd()
root_dir = os.path.join(cur_dir, "Dokumente", "Mülltrennung")

# Store PDFs in a list of dictionaries containing attributes for metadata
documents = [
    {"document_name": "FES_waskommtwohinein.pdf", "category": "mülltrennung_allgemein"},
    {"document_name": "FES_keinplastikindiebiotonne.pdf", "category": "mülltrennung_bio"},
    {"document_name": "MW_wertstofftonne.pdf", "category": "mülltrennung_wertstoff"}
]

preprocessed_docs = preprocess_docs(documents=documents, root_dir=root_dir)