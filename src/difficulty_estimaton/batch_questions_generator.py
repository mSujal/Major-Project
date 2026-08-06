
"""
Batch MCQ generator for data collection from system.

Usage:
    python batch_generate_mcq.py --pdf-dir /path/to/pdfs --output-dir mcq_output
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

import config
from src.doc_processing.late_chunking import LateChunking
from src.doc_processing.ragpipeline import RAGPipeline
from src.doc_processing.text_extraction import Extraction
from src.doc_processing.vector_store import VectorStore

load_dotenv(find_dotenv(usecwd=True))
API_KEY = os.getenv("GROQ_API_KEY")

PROJECT_ROOT = Path(__file__).parent

def process_pdf(pdf_path: str, rag: RAGPipeline, vs: VectorStore,
                 num_questions: int, output_dir: str):
    print(f"\n{'='*60}\n[PDF] {pdf_path}\n{'='*60}")

    if vs.is_indexed(pdf_path):
        print(f"[INFO] Already indexed (matched by file content) — skipping extraction.")
        rag.index(None, pdf_path)  # loads existing chunks from the store
    else:
        print("[Extraction] Starting...")
        pages = Extraction(pdf_path).extract_text()
        print(f"[Extraction] Done: {len(pages)} pages")

        print("[Index] Indexing...")
        rag.index(pages, pdf_path)

    print(f"[MCQ] Generating {num_questions} questions...")
    rag.query_mcq(
        question="all_topics",
        pdf_path=pdf_path,
        num_questions=num_questions,
        save_json=True,
        output_dir=output_dir,
    )
    print(f"[DONE] {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch-generate MCQs for all PDFs in a directory")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="mcq_output", help="Directory to save MCQ JSON files")
    parser.add_argument("--num", type=int, default=10, help="Number of questions per PDF (default 10)")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit(
            "[ERROR] GROQ_API_KEY not found. Create a .env file next to this script "
            "with a line: GROQ_API_KEY=your_key_here"
        )

    pdf_dir = Path(args.pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"[ERROR] No PDFs found in {pdf_dir}")

    print(f"[INFO] Found {len(pdf_files)} PDF(s) in {pdf_dir}")

    lc = LateChunking(model_name=config.MODEL, tokenizer_name=config.TOKENIZER)
    vs = VectorStore()
    rag = RAGPipeline(late_chunking=lc, api_key=API_KEY, vector_store=vs)

    os.makedirs(args.output_dir, exist_ok=True)

    succeeded, failed = [], []
    for pdf_path in pdf_files:
        try:
            process_pdf(str(pdf_path), rag, vs, args.num, args.output_dir)
            succeeded.append(str(pdf_path))
        except Exception as e:
            print(f"[ERROR] Failed on {pdf_path}: {e}")
            failed.append((str(pdf_path), str(e)))

    print(f"\n{'='*60}\n[SUMMARY]")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")
    for path, err in failed:
        print(f"    - {path}: {err}")


if __name__ == "__main__":
    main()
