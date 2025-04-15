#!/usr/bin/env python3

from bs4 import BeautifulSoup
import trafilatura  # Better than bs4 alone for main text extraction
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Optional,Dict
import re
import json
import os

from dotenv import load_dotenv
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disables parallelism warning of SentenceTransformer 

EMBEDDING_MODEL=os.getenv('EMBEDDING_MODEL')

class WebPageVectorDB:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

    def scrape_page(self, url: str) -> Optional[Dict]:
        """Extract clean text and metadata from web page"""
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
                print(f"Failed to download {url}")
                return None
        
        # Extract content (returns JSON string when successful)
        result = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                output_format='json'  # Returns JSON string
        )
        
        # Handle failed extraction
        if not result:
                print(f"Failed to extract content from {url}")
                return None
        
        try:
                # Parse JSON string to dictionary
                result_dict = json.loads(result)
                return {
                'url': url,
                'text': result_dict.get('text', ''),
                'title': result_dict.get('title', 'No title'),
                'date': result_dict.get('date', ''),
                'site_name': result_dict.get('sitename', '')
                }
        except json.JSONDecodeError:
                print(f"Invalid JSON from {url}")
                return None
    
    def preprocess_text(self, text: str) -> str:
        """Clean and chunk text for embeddings"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Simple chunking (for better approaches see below)
        max_chunk_length = 500
        words = text.split()
        chunks = [' '.join(words[i:i + max_chunk_length]) 
                for i in range(0, len(words), max_chunk_length)]
        return chunks
    
    def add_web_page(self, url: str):
        """Process and store a web page"""
        page_data = self.scrape_page(url)
        if not page_data:
            print(f"Failed to scrape {url}")
            return
            
        chunks = self.preprocess_text(page_data['text'])
        embeddings = self.embedding_model.encode(chunks)
        
        # Store with metadata
        ids = [f"{url}-chunk-{i}" for i in range(len(chunks))]
        metadatas = [{
            'url': page_data['url'],
            'title': page_data['title'],
            'chunk_num': i
        } for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(chunks)} chunks from {url}")

    def query(self, question: str, top_k: int = 3) -> List[dict]:
        """Search across all stored web pages"""
        query_embedding = self.embedding_model.encode([question]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return [{
            'content': doc,
            'metadata': meta,
            'score': 1 - dist  # Convert distance to similarity
        } for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )]

# Usage Example
if __name__ == "__main__":
    db = WebPageVectorDB()
    db.add_web_page("https://en.wikipedia.org/wiki/Artificial_intelligence")
    
    results = db.query("What are the main applications of AI?")
    for result in results:
        print(f"\nScore: {result['score']:.3f}")
        print(f"Source: {result['metadata']['title']} ({result['metadata']['url']})")
        print(f"Content: {result['content'][:200]}...")