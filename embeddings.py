#!/usr/bin/env python3

import os
import chromadb
from chromadb.config import Settings
import requests
from typing import List

from dotenv import load_dotenv
load_dotenv()


embedding_model=os.getenv('EMBEDDING_MODEL')


HF_API_URL = os.getenv('HF_API_EMBEDDING_MODEL_URL')
HF_TOKEN = os.getenv("HF_API_TOKEN")  # Set your token in environment variables


# chroma_client = chromadb.Client(Settings(allow_reset=True)) # Data vanishes when program ends
chroma_client = chromadb.PersistentClient(path="./chroma_db",  allow_reset=True)  # Data saved to disk
# chroma_client.reset()
# Create or get the collection
collection = chroma_client.get_or_create_collection(name="knowledge_base")

documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is one of the wonders of the world.",
    "Python is a popular programming language.",
    "OpenAI develops advanced AI models like GPT-4."
]
metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]
ids = [f"id_{i}" for i in range(len(documents))]



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
    print('populating chroma')
    # Get embeddings from OpenAI
    # Add to ChromaDB
    embeddings = get_hf_embeddings(documents)
    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings  # This is where your embeddings are inserted
)


def retrieve_documents(query, top_k=2):
    # Get query embedding
    query_embedding = get_hf_embeddings([query])[0]  # Get first embedding

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]

## FUNCTIONS USING GEMINI EMBEDDING MODEL
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



if __name__ == "__main__":
        populate_vector_db()
        print('chroma populated')