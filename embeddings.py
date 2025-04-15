#!/usr/bin/env python3

import os
import chromadb
from chromadb.config import Settings
import requests
from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np

from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disables parallelism warning of SentenceTransformer 

EMBEDDING_MODEL=os.getenv('EMBEDDING_MODEL')
embedding_model = SentenceTransformer(EMBEDDING_MODEL)  # Example lightweight model

chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Data saved to disk
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

def get_embeddings(documents):
    # Convert documents to embeddings
    embeddings = embedding_model.encode(documents, convert_to_tensor=False)
    return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings


def populate_vector_db():
    # get embeddings from hugging face
    embeddings = get_embeddings(documents)
    # Add to ChromaDB
    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings  # This is where your embeddings are inserted
)


def retrieve_documents(query, top_k=2):
    # Get query embedding
    query_embedding = embedding_model.encode([query]).tolist()[0]

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]



if __name__ == "__main__":
        populate_vector_db()
        print('chroma populated')
        query_embedding = embedding_model.encode([documents[0]]).tolist()[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        result = [{
            'content': doc,
            'metadata': meta,
            'score': 1 - dist  # Convert distance to similarity
        } for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )]

        print(result)



