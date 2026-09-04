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

# Resolve the actual project root (the folder containing config.py),
# not just the folder this script happens to live in. This lets the
# script run correctly whether it's placed at project root or nested
# under src/doc_processing/ (or anywhere else), and regardless of the
# caller's current working directory.
def _find_project_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "config.py").exists():
            return candidate
    # fallback: assume script sits under <root>/src/doc_processing/
    return start.parent.parent.parent if len(start.parents) >= 3 else start

PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import config
from src.doc_processing.text_extraction import Extraction
from src.doc_processing.late_chunking import LateChunking
from src.doc_processing.vector_store import VectorStore


def build_components():
    print("[init] Loading model and tokenizer...")
    lc = LateChunking(model_name=config.MODEL, tokenizer_name=config.TOKENIZER)
    vs = VectorStore(persist_dir=config.PERSIST_DIR)
    print("[init] Ready.\n")
    return lc, vs


def profile_pdf(pdf_path: str, lc: LateChunking, vs: VectorStore, force_fresh: bool) -> dict:
    pdf_path = str(pdf_path)

    result = {
        "pdf": Path(pdf_path).name,
        "pdf_path": pdf_path,
        "tier": Path(pdf_path).parent.name,  # large / medium / short / very_large
        "profiled_at": datetime.now().isoformat(),
        "mode": None,  # "fresh" or "cached"

        "page_count": None,
        "chunk_count": None,
        "token_count": None,
        "window_count": None,

        "time_extraction_s": None,
        "time_chunking_s": None,
        "time_tokenize_s": None,
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
            # NOTE: VectorStore.load() returns (chunks, pages) only — no
            # embeddings. A cache hit therefore gives us chunk/page stats
            # but NOT embeddings to reuse for retrieval. That's fine for
            # the indexing profiler (we're only measuring load speed +
            # chunk counts here), but flag this gap before wiring up
            # retrieval against cached runs.
            chunks, pages = vs.load(pdf_path)
            result["time_cache_load_s"] = round(time.perf_counter() - t0, 3)

            lc.chunks = chunks
            lc.chunk_pages = pages
            result["chunk_count"] = len(chunks)

            print(f"  [cache]   {len(chunks)} chunks loaded in {result['time_cache_load_s']}s")

        else:
            result["mode"] = "fresh"
            print(f"  [fresh]   not indexed — running full pipeline")

            # stage 1: extraction (instance-based per current text_extraction.py)
            t0 = time.perf_counter()
            pages = Extraction(pdf_path).extract_text()
            result["time_extraction_s"] = round(time.perf_counter() - t0, 3)
            result["page_count"] = len(pages)
            print(f"  [extract] {len(pages)} pages in {result['time_extraction_s']}s")

            corpus = "\n\n".join(text for _, text in pages)

            # stage 2: chunking
            t0 = time.perf_counter()
            lc._chunk(pages)
            result["time_chunking_s"] = round(time.perf_counter() - t0, 3)
            result["chunk_count"] = len(lc.chunks)
            print(f"  [chunk]   {len(lc.chunks)} chunks in {result['time_chunking_s']}s")

            # stage 3: tokenize full corpus
            t0 = time.perf_counter()
            input_ids, offset_mapping = lc._tokenize_full(corpus)
            result["time_tokenize_s"] = round(time.perf_counter() - t0, 3)
            T = input_ids.shape[1]
            n_windows = max(1, (T - lc.window_overlap - 1) // lc.window_stride + 1)
            result["token_count"] = T
            result["window_count"] = n_windows
            print(f"  [token]   {T} tokens / {n_windows} window(s) in {result['time_tokenize_s']}s")

            # stage 4: token boundaries + windowed embedding
            t0 = time.perf_counter()
            token_boundaries = lc._find_token_boundaries(corpus, offset_mapping)
            embeddings = lc._embed_windowed(input_ids, token_boundaries)
            lc.chunk_embeddings = embeddings
            result["time_embedding_s"] = round(time.perf_counter() - t0, 3)
            print(f"  [embed]   {len(embeddings)} embeddings in {result['time_embedding_s']}s")

            # stage 5: store
            t0 = time.perf_counter()
            vs.store(pdf_path, lc.chunks, embeddings, lc.chunk_pages)
            result["time_store_s"] = round(time.perf_counter() - t0, 3)
            print(f"  [store]   saved to ChromaDB in {result['time_store_s']}s")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"  [ERROR]   {result['error']}")
        traceback.print_exc()

    result["time_total_s"] = round(time.perf_counter() - total_start, 3)
    print(f"  [total]   {result['mode']} — {result['time_total_s']}s\n")
    return result


def write_csv(summary: list, out_dir: Path):
    """Flat CSV for direct import into the paper's indexing table."""
    fields = [
        "pdf", "tier", "mode", "page_count", "chunk_count", "token_count", "window_count",
        "time_extraction_s", "time_chunking_s", "time_tokenize_s",
        "time_embedding_s", "time_store_s", "time_cache_load_s",
        "time_total_s", "error",
    ]
    csv_path = out_dir / "indexing_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary:
            writer.writerow({k: r.get(k) for k in fields})
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="v2 indexing/extraction profiler — run BEFORE retrieval eval"
    )
    parser.add_argument("--pdf_dir", required=True,
                        help="Directory containing PDFs (searched recursively)")
    parser.add_argument("--out", default="profiling_results_v2",
                        help="Output directory for JSON/CSV (default: profiling_results_v2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only profile first N PDFs (smoke-test)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete any existing ChromaDB collection per PDF before "
                             "indexing, forcing a true from-scratch v2 run. Use this "
                             "for the real paper numbers — do NOT mix cached v1 "
                             "embeddings with fresh v2 ones.")
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
    if args.fresh:
        print("--fresh set: existing ChromaDB collections will be deleted and rebuilt.")
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

    tier_stats = {}
    for r in summary:
        if r["error"]:
            continue
        t = r.get("tier", "unknown")
        tier_stats.setdefault(t, []).append(r)

    def _avg(rows, key):
        vals = [row[key] for row in rows if row.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    tier_summary = {}
    for tier, rows in tier_stats.items():
        tier_summary[tier] = {
            "n_docs": len(rows),
            "avg_page_count": _avg(rows, "page_count"),
            "avg_chunk_count": _avg(rows, "chunk_count"),
            "avg_token_count": _avg(rows, "token_count"),
            "avg_window_count": _avg(rows, "window_count"),
            "avg_time_extraction_s": _avg(rows, "time_extraction_s"),
            "avg_time_chunking_s": _avg(rows, "time_chunking_s"),
            "avg_time_tokenize_s": _avg(rows, "time_tokenize_s"),
            "avg_time_embedding_s": _avg(rows, "time_embedding_s"),
            "avg_time_store_s": _avg(rows, "time_store_s"),
            "avg_time_total_s": _avg(rows, "time_total_s"),
        }

    summary_file = out_dir / "_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "profiled_at": datetime.now().isoformat(),
            "pdf_dir": str(pdf_dir),
            "n_pdfs": len(summary),
            "fresh_run": args.fresh,
            "tier_summary": tier_summary,
            "results": summary,
        }, f, indent=2)

    csv_path = write_csv(summary, out_dir)

    n_errors = sum(1 for r in summary if r["error"])

    w = 74
    print(f"{'═'*w}")
    print(f"  Results -> {out_dir}/  (+ {csv_path.name})")
    print(f"{'─'*w}")
    print(f"  {'PDF':<30} {'tier':>9} {'mode':>6} {'pg':>3} {'chk':>4} {'tok':>6} "
          f"{'win':>3} {'total(s)':>9}")
    print(f"  {'─'*30} {'─'*9} {'─'*6} {'─'*3} {'─'*4} {'─'*6} {'─'*3} {'─'*9}")

    for r in summary:
        name = r["pdf"][:30]
        tier = r.get("tier", "—")[:9]
        if r["error"]:
            print(f"  {name:<30} {tier:>9} ERROR: {r['error'][:30]}")
            continue
        mode = r["mode"]
        pages = str(r["page_count"]) if r["page_count"] else "—"
        chks = str(r["chunk_count"]) if r["chunk_count"] else "—"
        toks = str(r["token_count"]) if r["token_count"] else "—"
        wins = str(r["window_count"]) if r["window_count"] else "—"
        print(f"  {name:<30} {tier:>9} {mode:>6} {pages:>3} {chks:>4} {toks:>6} "
              f"{wins:>3} {r['time_total_s']:>9.2f}s")

    print(f"{'═'*w}")
    if n_errors:
        print(f"\n  ⚠ {n_errors}/{len(summary)} document(s) errored — check JSON files "
              f"and console output above before treating this run as final.")
    print(f"\n  Next: run retrieval eval script against this same v2 index.\n")


if __name__ == "__main__":
    main()