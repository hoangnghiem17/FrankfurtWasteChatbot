import os

import streamlit as st
from groq import Groq

from config import chroma_client

from dotenv import load_dotenv

load_dotenv()

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

def define_prompt(query, chat_history, relevant_passages):
  """
  Constructs a prompt for the chatbot by combining the user's query with relevant passages and the conversation history.

  Parameters:
  - query (str): The user's search query or question.
  - chat history (list of dict): A list of dictionaries representation conversation history, containing "user" and "chatbot".
  - relevant_passages (list): A list of relevant document passages retrieved from the Chroma collection.

  Returns:
  - str: A formatted prompt string that incorporates the user's query, relevant passages and conversation history.
  """
  processed_passages = [
    passage.replace("'", "").replace('"', "").replace("\n", " ")
    for passage in relevant_passages
  ]

  # Combine previous history with new query
  history_text = ""
  for entry in chat_history:
      history_text += f"User: {entry['user']}\nBot: {entry['chatbot']}\n"

  prompt = (
  f"""You are a knowledgeable and helpful chatbot specialized in waste management for residents of Frankfurt am Main. 
  You will answer questions in the same language in which they are asked. Use the provided context to inform your answer, 
  but do not rely on it verbatim—rephrase and integrate the information naturally into your response. If the context does not 
  directly apply to the question, generate an answer based on your understanding and provide helpful, relevant information. 
  Always aim to be comprehensive, clear, and accurate, reflecting local regulations and practices related to waste management 
  in Frankfurt am Main.
     
  Here is the conversation so far:
  {history_text}
        
  QUESTION: '{query}'
  CONTEXT: '{processed_passages}'

  ANSWER:
  """
    )
  
  return prompt

def query_groq_api(query, chat_history):
    """
    Queries the GROQ API with the constructed prompt to generate a response.

    Parameters:
    - query (str): The user's search query or question.
    - chat history (list of dict): A list of dictionaries representation conversation history, containing "user" and "chatbot".

    Returns:
    - tuple: A tuple containing:
        - str: The generated answer from the chatbot.
        - list: The list of relevant document passages used to generate the answer.
    """
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Perform similarity search to construct query and context 
    db = load_chroma_collection(name="frankfurt_waste_chatbot_v1")
    relevant_passages = get_relevant_passages(query=query,db=db,n_results=3)
    
    prompt = define_prompt(query=query, chat_history=chat_history, relevant_passages=relevant_passages)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ], 
        model="gemma-7b-it"
    )
    answer = chat_completion.choices[0].message.content

    return answer, relevant_passages

def get_user_input():
    """
    Prompts the user to input a query via a text input field in a Streamlit application.

    Returns:
    - str: The user's inputted query or question as a string.
    """
    return st.text_input("Ask a question:")

def generate_answer(user_question):
    """
    Generates answers by calling the GROQ API and updates the chat history and relevant references in a Streamlit application.

    Parameters:
    - user_question (str): The user's inputted query or question.

    Returns:
    - None
    """
    with st.spinner("Generating answer..."):
        answer, relevant_passages = query_groq_api(query=user_question, chat_history=st.session_state.chat_history)
        
    st.session_state.chat_history.append({"user": user_question, "chatbot": answer})
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("### Chat History")
        for entry in st.session_state.chat_history:
            st.write(f"**User:** {entry['user']}")
            st.write(f"**Chatbot:** {entry['chatbot']}")
        
    with col2:      
        st.write("### References Provided:")
        for i, passage in enumerate(relevant_passages, start=1):
            st.write(f"**Reference {i}:** {passage}")
            
# Main function to run the Streamlit app
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Frankfurt Waste Chatbot")
    st.write("Hello, I am a chatbot based on the LLM Gemma of Google. Ask me any questions about waste management in Frankfurt!")
  
    # Initialize session state for chat history if not already done
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # User input for the query
    user_question = get_user_input()
    
    if user_question:
        # Process the query and update the chat history
       generate_answer(user_question=user_question)

