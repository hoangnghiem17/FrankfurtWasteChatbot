import os

from groq import Groq

from config import groq_apikey
import querying_custom

def query_groq_api(prompt):
    client = Groq(
        api_key=groq_apikey
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt = define_prompt(query="Was sind die verschiedenen Mülltonnenarten in Frankfurt am Main?", relevant_passage=relevant_text)
            }
        ], 
        model="gemma-7b-it"
    )
    answer = chat_completion.choices[0].message.content
    
    return answer


    
