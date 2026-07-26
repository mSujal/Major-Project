import os 
import argparse
import config 
from time import time 
from pathlib import Path 
from dotenv import load_dotenv

from src.doc_processing.late_chunking import LateChunking
from src.doc_processing.ragpipeline import RAGPipeline
from src.doc_processing.text_extraction import Extraction
from src.doc_processing.vector_store import VectorStore

load_dotenv(Path(__file__).parent / ".env")
api_key = os.getenv("GROQ_API_KEY")

PORJECT_ROOT = Path(__file__).parent.parent 
DB_PATH = PORJECT_ROOT / "storage/chroma_db"

class Headless:
    def __init__(self, ragpipeline: RAGPipeline):
        self.current_pdf = None 
        self.rag_pipeline = ragpipeline
        self.pages = None

    def load_pdf(self, pdf_path: str):
        self.current_pdf = pdf_path 
        print("f[PDF] PDF set to : {[pdf_path]}")

    def extraction(self):
        print("\n\n[Extraction] Extraction Started")
        self.pages = Extraction.extract_text(self.current_pdf)
        print(f"[Extraction] Extraction Done : {len(self.pages)} pages")

    def index(self):
        print("\n\n[Index] Indexing started")
        self.rag_pipeline.index(self.pages, self.current_pdf)
    
    def generate_mcq(self, topic: str ="all topic", num_questions: int=5 , save_json: bool=False, output_dir: str="mcq_output"):
        print(f"[Generating] {num_questions} MCQS on {topic if topic != 'any topic' else 'General topics from PDF'}")
        results = self.rag_pipeline.query_mcq(
            question=topic, # need to make better prompt here (nudging in future)
            num_questions=num_questions,
            save_json=save_json,
            output_dir=output_dir
        )
        print(f"[Generation] Generated {num_questions} MCQS on {topic if topic != 'any topic' else 'General topics from PDF'}\n\t Stored in mcq_output directory")
        print(type(result))
        return result 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless MCQ Generator")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--topic", help="Topic to generate MCQ about")
    parser.add_argument("--num", type=int, default=5, help="Number of MCQS to generate. Default 5")
    parser.add_argument("--save", action="store_true", help="Save output as JSON")
    parser.add_argument("--output-dir", default="mcq_output", help="Output directory for storing MCQs")
    args = parser.parse_args()

    lc = LateChunking(
        model_name=config.MODEL,
        tokenizer_name=config.TOKENIZER
    )
    vs = VectorStore()
    ms = MCQStore()
    rag = RAGPipeline(
        late_chunking=lc,
        api_key=api_key,
        vector_store=vs,
        mcq_store=ms
    )
    
    headless = Headless(ragpipeline=rag)
    headless.load_pdf(args.pdf)

    if not vs.is_indexed(args.pdf):
        headless.extract()


    headless.index()
    result = headless.generate_mcq(
        topic=args.topic or "all_topics",
        num_questions=args.num, 
        save_json=args.save,
        output_dir=args.output_dir
    )


   # TODO: Check if this run creates and creates something and see how that can be feeded in layer 1 somehow.