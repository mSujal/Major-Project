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
# NOTE: MCQStore doesn't exist in the codebase yet, so we skip it for now.
# RAGPipeline already handles mcq_store=None fine (dedup/storage just gets skipped).

load_dotenv(Path(__file__).parent / ".env")
api_key = os.getenv("GROQ_API_KEY")

PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "storage/chroma_db"


class Headless:
    def __init__(self, ragpipeline: RAGPipeline):
        self.current_pdf = None
        self.rag_pipeline = ragpipeline
        self.pages = None

    def load_pdf(self, pdf_path: str):
        self.current_pdf = pdf_path
        print(f"[PDF] PDF set to: {pdf_path}")

    def extraction(self):
        print("\n\n[Extraction] Extraction Started")
        # Extraction needs the pdf path in __init__, then call extract_text() with no args
        self.pages = Extraction(self.current_pdf).extract_text()
        print(f"[Extraction] Extraction Done: {len(self.pages)} pages")

    def index(self):
        print("\n\n[Index] Indexing started")
        self.rag_pipeline.index(self.pages, self.current_pdf)

    def generate_mcq(self, pdf_path:str, topic: str = "all_topics", num_questions: int = 5,
                      save_json: bool = False, output_dir: str = "mcq_output"):
        label = topic if topic != "all_topics" else "General topics from PDF"
        print(f"[Generating] {num_questions} MCQs on {label}")
        result = self.rag_pipeline.query_mcq(
            question=topic,
            pdf_path=pdf_path,          # <-- added pdf_path
            num_questions=num_questions,
            save_json=save_json,
            output_dir=output_dir
        )
        print(f"[Generation] Generated {num_questions} MCQs on {label}\n\tStored in {output_dir}")
        return result

# Exposed function to call 
def generate_mcq_from_pdf(
    pdf_path: str,
    topic: str = 'all_topics',
    num_questions: int = 5,
    save_json: bool = False, 
    output_dir: str = "mcq_output"
):
    lc = LateChunking(model_name=config.MODEL, tokenizer_name=config.TOKENIZER)
    vs = VectorStore() 
    # ms = MCQStore() 
    rag = RAGPipeline(
        late_chunking=lc, 
        api_key=api_key,
        vector_store=vs,
        # mcq_store=ms
    ) 

    headless = Headless(ragpipeline=rag)
    headless.load_pdf(pdf_path=pdf_path)

    if not vs.is_indexed(pdf_path):
        headless.extraction()
        headless.index() 
    else:
        print(f"[INFO] PDF already indexed: {pdf_path}")

    results = headless.generate_mcq(
        topic=topic, 
        pdf_path=pdf_path,
        num_questions=2,#num_questions,
        save_json=save_json,
        output_dir=output_dir
    )
    print(results)
    return results



#### below this is redundant to the exposed function 

if __name__ == "__main__":
    generate_mcq_from_pdf(pdf_path="/home/sujal/Downloads/Full_Concept_Note_DocuMind_Adaptive_MCQ_System.pdf")
    # parser = argparse.ArgumentParser(description="Headless MCQ Generator")
    # parser.add_argument("--pdf", required=True, help="Path to PDF file")
    # parser.add_argument("--topic", help="Topic to generate MCQ about")
    # parser.add_argument("--num", type=int, default=5, help="Number of MCQs to generate. Default 5")
    # parser.add_argument("--save", action="store_true", help="Save output as JSON")
    # parser.add_argument("--output-dir", default="mcq_output", help="Output directory for storing MCQs")
    # args = parser.parse_args()

    # if not api_key:
    #     raise SystemExit(
    #         "[ERROR] GROQ_API_KEY not found. Create a .env file next to this script "
    #         "with a line: GROQ_API_KEY=your_key_here"
    #     )

    # lc = LateChunking(
    #     model_name=config.MODEL,
    #     tokenizer_name=config.TOKENIZER
    # )
    # vs = VectorStore()
    # rag = RAGPipeline(
    #     late_chunking=lc,
    #     api_key=api_key,
    #     vector_store=vs,
    #     mcq_store=None  # not implemented yet — safe to skip for now
    # )

    # headless = Headless(ragpipeline=rag)
    # headless.load_pdf(args.pdf)

    # if not vs.is_indexed(args.pdf):
    #     headless.extraction()

    # headless.index()

    # if args.save:
    #     os.makedirs(args.output_dir, exist_ok=True)

    # result = headless.generate_mcq(
    #     topic=args.topic or "all_topics",
    #     num_questions=args.num,
    #     save_json=args.save,
    #     output_dir=args.output_dir
    # )

    # print(result)