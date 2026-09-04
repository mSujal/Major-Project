"""
Temporary script to create the data for evaluation

    1. Embed all documents (before running) 
    2. Run each query as latechunking, naive chunking and bm25
    3. dump output per query for each method in json file
"""

import argparse 
from time import time 
import os 

import config 
from src.doc_processing.late_chunking import LateChunking
from src.doc_processing.ragpipeline import RAGPipeline
from src.doc_processing.vector_store import VectorStore
from src.doc_processing.text_extraction import Extraction


from pathlib import Path 
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
api_key = os.getenv("GROQ_API_KEY")

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "chromadb"

class Headless():
    def __ini__(self, ragpipeline):
        self.current_pdf = None 
        self.ragpipeline = ragpipeline
        self.pages = None 

    def load_pdf(self, )