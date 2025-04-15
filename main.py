#!/usr/bin/env python3

import os
from openai import OpenAI

from embeddings import retrieve_documents 

from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
gemini_api_url = os.getenv('GEMINI_API_URL')


model = os.getenv('MODEL')


client = OpenAI(
            api_key=gemini_api_key,  # Google Gemini API key
            base_url=gemini_api_url,  # Gemini base URL
)



def generate_answer(query):
    # Retrieve relevant documents
    relevant_docs = retrieve_documents(query)
    context = "\n".join(relevant_docs)
    
    # Generate response
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Answer the question based on this context:\n{context}"},
            {"role": "user", "content": query}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content



# Example usage:
if __name__ == "__main__":
        
        queries = [
                "What is the capital of France?",
                "Where is the Eiffel Tower located?",
                "What does OpenAI develop?"
        ]

        for query in queries:
                print(f"\nQuestion: {query}")
                answer = generate_answer(query)
                print(f"Answer: {answer}")

