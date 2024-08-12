from config import chroma_client

#-----------STATE: Documents are preprocessed, chunked, embedded and stored in Chroma vector store.
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

def get_relevant_passages(query, db, n_results):
  """
  Retrieves the most relevant documents from the Chroma collection based on the given query.

  Parameters:
  - query (str): The search query used to find relevant documents in the collection.
  - db (chromadb.Collection): The Chroma Collection from which to retrieve documents.
  - n_results (int): The number of top results to return based on relevance.

  Returns:
  - list: The most relevant documents corresponding to the query.
  """
  passages = db.query(query_texts=[query], n_results=n_results)['documents'][0]
  
  return passages

def define_prompt(query, relevant_passages):
  processed_passages = [
    passage.replace("'", "").replace('"', "").replace("\n", " ")
    for passage in relevant_passages
  ]
  
  prompt = ("""You are a helpful and informative chatbot about waste separation that answers questions using text from the reference passage included below. \
  Be sure to respond in the language in which the question is asked in a complete sentence, being comprehensive, including all relevant background information. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passages}'

  ANSWER:
  """).format(query=query, relevant_passages=processed_passages)
  
  return prompt
 
from groq import Groq

from config import groq_apikey

def query_groq_api(query):
    client = Groq(
        api_key=groq_apikey
    )

    # Perform similarity search to construct query and context
    db = load_chroma_collection(name="frankfurt_waste_chatbot_v1")
    relevant_passages = get_relevant_passages(query=query,db=db,n_results=3)
    
    """
    # Print each relevant text in a separate line with its index
    print(type(relevant_text))
    for index, text in enumerate(relevant_text):
        print(f"{index + 1}\n {text}")
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": define_prompt(query=query, relevant_passages=relevant_passages)
            }
        ], 
        model="gemma-7b-it"
    )
    answer = chat_completion.choices[0].message.content
    
    return answer

query = "Was sind die verschiedenen Mülltonnenarten in Frankfurt am Main?"
answer = query_groq_api(query=query)
print(answer)
    

