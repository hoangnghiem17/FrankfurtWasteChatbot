import os
import logging
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader

from config import document_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def correct_ger_umlauts(text: str) -> str:
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


def preprocess_docs(documents: List[Dict[str,str]], root_dir: str) -> List:
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

    # Validate input
    if not isinstance(documents, list) or not all(isinstance(doc, dict) for doc in documents):
        logging.error("Invalid input: 'documents' should be a list of dictionaries.")
        raise ValueError("Invalid input: 'documents' should be a list of dictionaries.")
    
    preprocessed_docs = []

    for doc_info in documents:
        pdf_path = os.path.join(document_directory, doc_info["document_name"])
        
        if not os.path.isfile(pdf_path):
            logging.warning(f"File not found: {pdf_path}. Skipping document.")
            continue

        try:
            # PyPDFLoader separates a document by page - access extracted text (page_content) or metadata (metadata)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            valid_docs = []

            for doc in docs:
                clean_text = correct_ger_umlauts(doc.page_content)
                word_count = len(clean_text.split())

                if word_count > 10:
                    doc.metadata["category"] = doc_info.get("category", "unknown")
                    doc.metadata["document_name"] = doc_info.get("document_name", "unknown")
                    valid_docs.append(doc)

            logging.info(f"Processed document: {doc_info['document_name']} with {len(valid_docs)} valid pages.")
            preprocessed_docs.extend(valid_docs)

        except Exception as e:
            logging.error(f"Error processing document {doc_info.get('document_name', 'unknown')}: {e}")
    
    return preprocessed_docs