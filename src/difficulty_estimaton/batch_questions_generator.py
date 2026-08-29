"""
Batch MCQ generator for data collection from system.

Usage:
    # Process all PDFs in a directory
    python -m src.difficulty_estimaton.batch_questions_generator --pdf-dir /path/to/pdfs --output-dir mcq_output

    # Process a single PDF file
    python -m src.difficulty_estimaton.batch_questions_generator --pdf-dir /path/to/single.pdf --output-dir mcq_output

    # Generate questions for a specific topic only
    python -m src.difficulty_estimaton.batch_questions_generator --pdf-dir /path/to/pdfs --topic cmmi

    # Generate questions for ALL topics (default if no --topic given)
    python -m src.difficulty_estimaton.batch_questions_generator --pdf-dir /path/to/pdfs
"""

import os
import argparse
from pathlib import Path
from typing import List
from dotenv import load_dotenv, find_dotenv

import config
from src.doc_processing.late_chunking import LateChunking
from src.doc_processing.ragpipeline import RAGPipeline
from src.doc_processing.text_extraction import Extraction
from src.doc_processing.vector_store import VectorStore

load_dotenv(find_dotenv(usecwd=True))
API_KEY = os.getenv("GROQ_API_KEY")

PROJECT_ROOT = Path(__file__).parent


def get_pdf_paths(input_path: str) -> List[Path]:
    """
    Returns a list of PDF Path objects:
    - If input_path is a file: returns that file if it's a PDF.
    - If input_path is a directory: returns all .pdf files inside it.
    - Otherwise raises a clear error.
    """
    path = Path(input_path)

    if path.is_file():
        if path.suffix.lower() == ".pdf":
            return [path]
        raise ValueError(f"File is not a PDF: {path}")

    if path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))
        if not pdfs:
            raise ValueError(f"No PDF files found in directory: {path}")
        return pdfs

    raise FileNotFoundError(f"Path does not exist: {path}")


def process_pdf(
    pdf_path: str,
    rag: RAGPipeline,
    vs: VectorStore,
    num_questions: int,
    output_dir: str,
    topic: str = None,
):
    """
    Process a single PDF: index it (if needed) and generate MCQs.
    If topic is None, generate for all topics in the taxonomy.
    """
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

    # Determine which topics to generate for
    if topic is not None:
        # Use the specified topic if it exists in taxonomy, else fallback
        if topic in rag.taxonomy["topics"]:
            topics_to_generate = [topic]
        else:
            print(f"[WARN] Topic '{topic}' not found in taxonomy. Using all topics.")
            topics_to_generate = list(rag.taxonomy["topics"].keys())
    else:
        # Generate for all topics
        topics_to_generate = list(rag.taxonomy["topics"].keys())

    print(f"[MCQ] Generating {num_questions} questions for {len(topics_to_generate)} topic(s)")

    for topic_name in topics_to_generate:
        print(f"  → Topic: {topic_name}")
        rag.query_mcq(
            query=topic_name,           # ✅ correct keyword
            pdf_path=pdf_path,
            num_questions=num_questions,
            save_json=True,
            output_dir=output_dir,
        )

    print(f"[DONE] {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate MCQs for PDFs (supports single file or directory)"
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="Path to a single PDF file OR a directory containing PDF files"
    )
    parser.add_argument(
        "--output-dir",
        default="mcq_output",
        help="Directory to save MCQ JSON files"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of questions per topic (default 10)"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Specific topic to generate questions for (e.g., 'testing', 'cmmi'). If not given, generates for all topics."
    )
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit(
            "[ERROR] GROQ_API_KEY not found. Create a .env file with: GROQ_API_KEY=your_key_here"
        )

    # Get list of PDF files (supports file or directory)
    try:
        pdf_files = get_pdf_paths(args.pdf_dir)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"[ERROR] {e}")

    print(f"[INFO] Found {len(pdf_files)} PDF(s) to process")

    # Initialize components
    lc = LateChunking(model_name=config.MODEL, tokenizer_name=config.TOKENIZER)
    vs = VectorStore()
    rag = RAGPipeline(late_chunking=lc, api_key=API_KEY, vector_store=vs)

    os.makedirs(args.output_dir, exist_ok=True)

    succeeded, failed = [], []
    for pdf_path in pdf_files:
        try:
            process_pdf(
                pdf_path=str(pdf_path),
                rag=rag,
                vs=vs,
                num_questions=args.num,
                output_dir=args.output_dir,
                topic=args.topic,
            )
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