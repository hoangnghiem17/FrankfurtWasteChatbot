import requests

from config import gemma_token, api_url

def query_llm(data):
    """
    Calls Gemma API for inference.

    Args:
        - data (dict): A json string as input query to LLM.
        
    Returns:
        - response (dict): A dictionary containing model answer.
    """

    # Headers including the authorization token
    headers = {"Authorization": f"Bearer {gemma_token}"}

    # Make the API request
    response = requests.post(api_url, headers=headers, json=data)
    
    return response.json()

#data = {"inputs": "Write me a poem about Machine Learning and AI."}
#print(query_llm(data=data))
