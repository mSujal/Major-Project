
import re 
import os
import sys
import json
import csv
import time
import platform
import argparse
import traceback
from pathlib import Path
from datetime import datetime

if platform.system() == "Windows":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _find_project_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "config.py").exists():
            return candidate
    return start.parent.parent.parent if len(start.parents) >= 3 else start

PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import torch
import config
from src.doc_processing.text_extraction import Extraction
from src.doc_processing.late_chunking import LateChunking
from src.doc_processing.vector_store import VectorStore

NAIVE_PERSIST_DIR = str(Path(config.PERSIST_DIR).parent / "chroma_db_naive")


def is_noise_chunk(chunk: str) -> bool:
    lines = [l.strip() for l in chunk.strip().split("\n") if l.strip()]
    if not lines:
        return True
    toc_lines = sum(
        1 for l in lines
        if re.search(r"(\.\s*){3,}|\bSection\b.+\d+$|\d+\.\d+.+\d+$", l)
    )
    if toc_lines / len(lines) > 0.3:
        return True
    if len(chunk.split()) < 30:
        return True
    return False


def build_components():
    print("[init] Loading model and tokenizer...")
    lc = LateChunking(model_name=config.MODEL, tokenizer_name=config.TOKENIZER)
    vs = VectorStore(persist_dir=NAIVE_PERSIST_DIR)
    print(f"[init] Naive index will be stored at: {NAIVE_PERSIST_DIR}")
    print("[init] Ready.\n")
    return lc, vs


def embed_chunks_independently(lc: LateChunking, chunks: list) -> list:
    embeddings = []
    for chunk in chunks:
        prefixed = "search_document: " + chunk
        tokens = lc.tokenizer(
            prefixed,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        )
        tokens = {k: v.to(lc.device) for k, v in tokens.items()}
        with torch.inference_mode():
            outputs = lc.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
        emb = outputs.last_hidden_state[0].mean(dim=0)
        embeddings.append(emb)
    return embeddings


def profile_pdf(pdf_path: str, lc: LateChunking, vs: VectorStore, force_fresh: bool) -> dict:
    pdf_path = str(pdf_path)

    result = {
        "pdf": Path(pdf_path).name,
        "pdf_path": pdf_path,
        "tier": Path(pdf_path).parent.name,
        "profiled_at": datetime.now().isoformat(),
        "mode": None,

        "page_count": None,
        "chunk_count": None,

        "time_extraction_s": None,
        "time_chunking_s": None,
        "time_embedding_s": None,
        "time_store_s": None,
        "time_cache_load_s": None,

        "time_total_s": None,
        "error": None,
    }

    total_start = time.perf_counter()

    try:
        if force_fresh and vs.is_indexed(pdf_path):
            vs.delete(pdf_path)

        if vs.is_indexed(pdf_path):
            result["mode"] = "cached"
            print(f"  [cache]   already indexed — loading from ChromaDB")
            t0 = time.perf_counter()
            chunks, pages = vs.load(pdf_path)
            result["time_cache_load_s"] = round(time.perf_counter() - t0, 3)
            result["chunk_count"] = len(chunks)
            print(f"  [cache]   {len(chunks)} chunks loaded in {result['time_cache_load_s']}s")

        else:
            result["mode"] = "fresh"
            print(f"  [fresh]   not indexed — running naive pipeline")

            # stage 1: extraction (identical to windowed system)
            t0 = time.perf_counter()
            pages = Extraction(pdf_path).extract_text()
            result["time_extraction_s"] = round(time.perf_counter() - t0, 3)
            result["page_count"] = len(pages)
            print(f"  [extract] {len(pages)} pages in {result['time_extraction_s']}s")

            # noise filtering: match RAGPipeline.index(), which filters
            # pages BEFORE chunking, not chunks after
            pages = [(p, t) for p, t in pages if not is_noise_chunk(t)]

            # stage 2: chunking — SAME splitter/params as windowed system
            # (RecursiveCharacterTextSplitter, same chunk_size/overlap),
            # only the embedding step differs from here on
            t0 = time.perf_counter()
            lc._chunk(pages)
            result["time_chunking_s"] = round(time.perf_counter() - t0, 3)
            result["chunk_count"] = len(lc.chunks)
            print(f"  [chunk]   {len(lc.chunks)} chunks in {result['time_chunking_s']}s")

            # stage 3: NAIVE embedding — independent per-chunk, no window pooling
            t0 = time.perf_counter()
            embeddings = embed_chunks_independently(lc, lc.chunks)
            lc.chunk_embeddings = embeddings
            result["time_embedding_s"] = round(time.perf_counter() - t0, 3)
            print(f"  [embed]   {len(embeddings)} embeddings (independent) in {result['time_embedding_s']}s")

            # stage 4: store to the SEPARATE naive persist dir
            t0 = time.perf_counter()
            vs.store(pdf_path, lc.chunks, embeddings, lc.chunk_pages)
            result["time_store_s"] = round(time.perf_counter() - t0, 3)
            print(f"  [store]   saved to ChromaDB (naive) in {result['time_store_s']}s")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"  [ERROR]   {result['error']}")
        traceback.print_exc()

    result["time_total_s"] = round(time.perf_counter() - total_start, 3)
    print(f"  [total]   {result['mode']} — {result['time_total_s']}s\n")
    return result


def write_csv(summary: list, out_dir: Path):
    fields = [
        "pdf", "tier", "mode", "page_count", "chunk_count",
        "time_extraction_s", "time_chunking_s", "time_embedding_s",
        "time_store_s", "time_cache_load_s", "time_total_s", "error",
    ]
    csv_path = out_dir / "indexing_summary_naive.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary:
            writer.writerow({k: r.get(k) for k in fields})
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Naive-chunking indexer — same splitter/model as the windowed "
                     "system, but each chunk embedded independently (no window pooling). "
                     "Stores to a SEPARATE ChromaDB dir so it never collides with the "
                     "windowed-system index."
    )
    parser.add_argument("--pdf_dir", required=True)
    parser.add_argument("--out", default="profiling_results_naive")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if args.limit:
        pdfs = pdfs[:args.limit]

    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    print(f"Found {len(pdfs)} PDF(s) in {pdf_dir}")
    print(f"Naive index dir: {NAIVE_PERSIST_DIR}")
    if args.fresh:
        print("--fresh set: existing naive ChromaDB collections will be deleted and rebuilt.")
    print("─" * 60)

    lc, vs = build_components()

    summary = []
    for pdf_path in pdfs:
        print(f"[doc] {pdf_path.name}")
        result = profile_pdf(str(pdf_path), lc, vs, force_fresh=args.fresh)
        out_file = out_dir / f"{pdf_path.stem}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        summary.append(result)

    summary_file = out_dir / "_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "profiled_at": datetime.now().isoformat(),
            "pdf_dir": str(pdf_dir),
            "naive_persist_dir": NAIVE_PERSIST_DIR,
            "n_pdfs": len(summary),
            "fresh_run": args.fresh,
            "results": summary,
        }, f, indent=2)

    csv_path = write_csv(summary, out_dir)
    n_errors = sum(1 for r in summary if r["error"])

    w = 74
    print(f"{'═'*w}")
    print(f"  Results -> {out_dir}/  (+ {csv_path.name})")
    print(f"  Naive index -> {NAIVE_PERSIST_DIR}")
    print(f"{'─'*w}")
    print(f"  {'PDF':<30} {'tier':>9} {'mode':>6} {'pg':>3} {'chk':>4} {'total(s)':>9}")
    print(f"  {'─'*30} {'─'*9} {'─'*6} {'─'*3} {'─'*4} {'─'*9}")

    for r in summary:
        name = r["pdf"][:30]
        tier = r.get("tier", "—")[:9]
        if r["error"]:
            print(f"  {name:<30} {tier:>9} ERROR: {r['error'][:30]}")
            continue
        mode = r["mode"]
        pages = str(r["page_count"]) if r["page_count"] else "—"
        chks = str(r["chunk_count"]) if r["chunk_count"] else "—"
        print(f"  {name:<30} {tier:>9} {mode:>6} {pages:>3} {chks:>4} {r['time_total_s']:>9.2f}s")

    print(f"{'═'*w}")
    if n_errors:
        print(f"\n  ⚠ {n_errors}/{len(summary)} document(s) errored — check before treating this run as final.")
    print(f"\n  Next: run the retrieval script with --method naive against this index,\n"
          f"  and --method bm25 (no index needed, built at query time from raw chunks).\n")


if __name__ == "__main__":
    main()