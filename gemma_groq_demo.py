import os

import streamlit as st
from groq import Groq

from config import chroma_client

from dotenv import load_dotenv

load_dotenv()

#-----------STATE: Documents are preprocessed, chunked, embedded and stored in Chroma vector store.

# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
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

def define_prompt(query, relevant_passages, chat_history):
  """
  Constructs a prompt for the chatbot by combining the user's query with relevant passages.

  Parameters:
  - query (str): The user's search query or question.
  - relevant_passages (list): A list of relevant document passages retrieved from the Chroma collection.

  Returns:
  - str: A formatted prompt string that incorporates the user's query and the relevant passages, designed to guide the chatbot in generating a comprehensive and context-aware response.
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
    Queries the GROQ API with the user's query and relevant document passages to generate a response.

    Parameters:
    - query (str): The user's search query or question.

    Returns:
    - tuple: A tuple containing the generated answer (str) and the list of relevant document passages (list) used to generate the answer.
    """
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Perform similarity search to construct query and context
    db = load_chroma_collection(name="frankfurt_waste_chatbot_v1")
    relevant_passages = get_relevant_passages(query=query,db=db,n_results=3)
      
    prompt = define_prompt(query=query, relevant_passages=relevant_passages, chat_history=chat_history)
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

# Create Streamlit app
st.title("Frankfurt Waste Chatbot")
st.write("Hello, I am a chatbot based on the LLM Gemma of Google. Ask me any questions about waste management in Frankfurt!")

# User input for the query
user_question = st.text_input("Ask a question:")

if user_question:
    # Query GROQ API with user question
    with st.spinner("Generating answer..."):
        answer, context = query_groq_api(query=user_question, chat_history=st.session_state.chat_history)
        
    # Update chat history
    st.session_state.chat_history.append({"user": user_question, "chatbot": answer})
        
    # Display ongoing chat history
    st.write("### Chat History")
    for entry in st.session_state.chat_history:
        st.write(f"**User:** {entry['user']}")
        st.write(f"**Chatbot:** {entry['chatbot']}")
        
    # Display the answer
    #st.write("### Latest Answer:")
    #st.write(answer)
    
    # Display relevant, concatenated document chunks used for answer generation
    st.write("### References Used:")
    for i, passage in enumerate(context, start=1):
        st.write(f"**Reference {i}:** {passage}")