#!/usr/bin/env python3

import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from tqdm import tqdm  # Progress bar
import os

class PDFEmbedder:
    def __init__(self, persist_dir: str = "./chroma_db"):
        # Initialize embedding model (local execution)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configure ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

    def extract_text(self, pdf_path: str) -> str:
        """Extract text with layout awareness using PyMuPDF"""
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text("text") for page in doc)

    def clean_text(self, text: str) -> str:
        """Fix common PDF artifacts"""
        text = re.sub(r'-\n(\w)', r'\1', text)  # Hyphenated words
        text = re.sub(r'\s+', ' ', text)        # Extra whitespace
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Semantic chunking with overlap (sliding window)"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def process_pdf(self, pdf_path: str, metadata: Optional[Dict] = None) -> int:
        """Process a single PDF and return chunk count"""
        try:
            raw_text = self.extract_text(pdf_path)
            clean_text = self.clean_text(raw_text)
            chunks = self.chunk_text(clean_text)
            
            # Batch embeddings for memory efficiency
            embeddings = []
            for i in range(0, len(chunks), 32):  # 32 chunks per batch
                batch = chunks[i:i+32]
                embeddings.extend(self.model.encode(batch, show_progress_bar=False))
            
            # Prepare metadata
            metadatas = [{**(metadata or {}), 
                         "source": os.path.basename(pdf_path),
                         "chunk_num": i} for i in range(len(chunks))]
            
            # Store in ChromaDB
            self.collection.add(
                documents=chunks,
                embeddings=[e.tolist() for e in embeddings],
                metadatas=metadatas,
                ids=[f"{os.path.basename(pdf_path)}-{i}" for i in range(len(chunks))]
            )
            return len(chunks)
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return 0

    def batch_process(self, pdf_paths: List[str], metadata_fn: Optional[callable] = None):
        """Process multiple PDFs with progress tracking"""
        total_chunks = 0
        for path in tqdm(pdf_paths, desc="Processing PDFs"):
            # Generate metadata if function provided
            metadata = metadata_fn(path) if metadata_fn else None
            total_chunks += self.process_pdf(path, metadata)
        print(f"Processed {len(pdf_paths)} PDFs ({total_chunks} total chunks)")

    def query(self, question: str, top_k: int = 3, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Semantic search with metadata filtering"""
        query_embedding = self.model.encode(question).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        return [{
            "content": doc,
            "score": 1 - dist,
            "metadata": meta,
            "source": meta.get("source")
        } for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )]

# Usage Example
if __name__ == "__main__":
    # Initialize embedder
    embedder = PDFEmbedder()
    
    # Example metadata extractor
    def get_metadata(pdf_path: str) -> Dict:
        return {
            "title": os.path.basename(pdf_path),
            "year": "2023"  # Extract from filename/path if needed
        }
    
    # Batch process all PDFs in a directory
    import glob
    pdf_files = glob.glob("./documents/*.pdf")
    
    # Process with metadata extraction
    embedder.batch_process(pdf_files, metadata_fn=get_metadata)
    
    # Query the collection
    results = embedder.query("What is stoicism ?", top_k=2)
    for res in results:
        print(f"\n[Score: {res['score']:.2f}] {res['metadata']['title']}")
        print(res['content'][:200] + "...")