#!/usr/bin/env python3

import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import requests
from typing import List

from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
gemini_api_url = os.getenv('GEMINI_API_URL')



model = os.getenv('MODEL')
embedding_model=os.getenv('EMBEDDING_MODEL')


HF_API_URL = os.getenv('HF_API_EMBEDDING_MODEL_URL')
HF_TOKEN = os.getenv("HF_API_TOKEN")  # Set your token in environment variables


client = OpenAI(
            api_key=gemini_api_key,  # Google Gemini API key
            base_url=gemini_api_url,  # Gemini base URL
)
chroma_client = chromadb.Client(Settings(allow_reset=True))

documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is one of the wonders of the world.",
    "Python is a popular programming language.",
    "OpenAI develops advanced AI models like GPT-4."
]
metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]
ids = [f"id_{i}" for i in range(len(documents))]


# Create or get the collection
collection = chroma_client.create_collection(name="knowledge_base")

def get_hf_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Hugging Face free API"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(
        HF_API_URL + "/" + embedding_model,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )
    return response.json()

# Step 1: Generate embeddings and store in ChromaDB
def populate_vector_db():
    # Get embeddings from OpenAI
    # Add to ChromaDB
    embeddings = get_hf_embeddings(documents)
    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings  # This is where your embeddings are inserted
)
# def populate_vector_db():
#     # Get embeddings from OpenAI
#     embeddings = client.embeddings.create(
#         input=documents,
#         model=embedding_model
#     ).data
    
#     # Convert to list of lists
#     embeddings_list = [e.embedding for e in embeddings]
    
#     # Add to ChromaDB
#     collection.add(
#         embeddings=embeddings_list,
#         documents=documents,
#         metadatas=metadata,
#         ids=ids
#     )



# Step 2: Retrieve relevant documents from ChromaDB
def retrieve_documents(query, top_k=2):
    # Get query embedding
    query_embedding = get_hf_embeddings([query])[0]  # Get first embedding

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    return results['documents'][0]
# def retrieve_documents(query, top_k=2):
#     # Get query embedding
#     query_embedding = client.embeddings.create(
#         input=[query],
#         model=embedding_model
#     ).data[0].embedding
    
#     # Query ChromaDB
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k
#     )
    
#     return results['documents'][0]


# Step 3: Generate answer using retrieved context
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
        
        populate_vector_db()
        queries = [
                "What is the capital of France?",
                "Where is the Eiffel Tower located?",
                "What does OpenAI develop?"
        ]

        for query in queries:
                print(f"\nQuestion: {query}")
                answer = generate_answer(query)
                print(f"Answer: {answer}")

        chroma_client.reset()
