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
import torch.nn.functional as F
import config
from src.doc_processing.late_chunking import LateChunking
from src.doc_processing.vector_store import VectorStore

try:
    from bm25_retriever import BM25Retriever
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bm25_retriever import BM25Retriever


def _resolve_dir(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


METHODS = ["windowed", "naive", "bm25"]


def embed_text(lc: LateChunking, text: str) -> torch.Tensor:
    """
    Mean‑pool the last hidden states of a piece of text.
    The same prefix ('search_query:') is used for all texts to keep the
    embedding space consistent (the model was trained with that prefix).
    """
    prefixed = "search_query: " + text
    tokens = lc.tokenizer(
        prefixed,
        return_tensors="pt",
        return_offsets_mapping=False,
        truncation=True,
        max_length=8192,
    )
    tokens = {k: v.to(lc.device) for k, v in tokens.items()}
    with torch.inference_mode():
        outputs = lc.model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
    return outputs.last_hidden_state[0].mean(dim=0)


def retrieve_dense(lc: LateChunking, chunks, chunk_pages, chunk_embeddings, query: str, top_k: int):
    """
    Dense retrieval using pre‑computed chunk embeddings.
    Returns (chunk_text, page_info, similarity_score).
    """
    query_emb = embed_text(lc, query)
    sims = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        if not isinstance(chunk_emb, torch.Tensor):
            chunk_emb = torch.tensor(chunk_emb, dtype=torch.float32).to(lc.device)
        score = F.cosine_similarity(query_emb.unsqueeze(0), chunk_emb.unsqueeze(0)).item()
        sims.append((score, i))
    sims.sort(reverse=True)
    top = sims[:top_k]
    # Return only the text, page, and score – we don't need the index any more
    return [(chunks[i], chunk_pages[i], round(score, 4)) for score, i in top]


def find_pdf(pdf_dir: Path, filename: str):
    matches = list(pdf_dir.rglob(filename))
    return matches[0] if matches else None


def run_eval(method: str, eval_set: list, pdf_dir: Path, lc: LateChunking,
             vs_windowed: VectorStore, vs_naive: VectorStore, top_k: int,
             similarity_threshold: float = 0.7):
    """
    Evaluate one retrieval method.
    Relevance is determined by cosine similarity between the reference_answer
    and the retrieved chunk (both embedded in isolation).
    """
    results = []
    cache = {"pdf_path": None, "chunks": None, "pages": None, "embeddings": None, "bm25": None}

    for item in eval_set:
        found = find_pdf(pdf_dir, item["document"])
        if found is None:
            print(f"  [skip] {item['document']} not found under {pdf_dir}")
            continue
        pdf_path = str(found)

        # Load data for this PDF if not already cached
        if pdf_path != cache["pdf_path"]:
            print(f"\n  [doc] {item['document']}  (method={method})")
            vs = vs_windowed if method == "windowed" else vs_naive

            if method in ("windowed", "naive"):
                if not vs.is_indexed(pdf_path):
                    print(f"    [warn] not indexed in {method} store — skipping doc")
                    cache["pdf_path"] = pdf_path
                    cache["chunks"] = []
                    cache["pages"] = []
                    cache["embeddings"] = []
                    cache["bm25"] = None
                else:
                    chunks, pages = vs.load(pdf_path)
                    col = vs._get_or_create_collection(pdf_path)
                    raw = col.get(include=["documents", "metadatas", "embeddings"])
                    combined = sorted(
                        zip(raw["metadatas"], raw["documents"], raw["embeddings"]),
                        key=lambda x: x[0]["chunk_index"],
                    )
                    _, _, embeddings = zip(*combined) if combined else ([], [], [])
                    cache["pdf_path"] = pdf_path
                    cache["chunks"] = chunks
                    cache["pages"] = pages
                    cache["embeddings"] = list(embeddings)
                    cache["bm25"] = None
                    print(f"    loaded {len(chunks)} chunks + embeddings from {method} store")

            elif method == "bm25":
                if not vs_naive.is_indexed(pdf_path):
                    print(f"    [warn] not indexed in naive store — BM25 needs it for chunk text — skipping doc")
                    cache["pdf_path"] = pdf_path
                    cache["bm25"] = None
                else:
                    chunks, pages = vs_naive.load(pdf_path)
                    cache["pdf_path"] = pdf_path
                    cache["bm25"] = BM25Retriever(chunks, pages)
                    print(f"    built BM25 index over {len(chunks)} naive chunks")

        query = item["question"]
        reference_answer = item.get("reference_answer", "")
        correct_page = item.get("correct_page")

        # Retrieve top‑k
        t0 = time.perf_counter()
        try:
            if method in ("windowed", "naive"):
                if not cache["chunks"]:
                    scored = []
                else:
                    scored = retrieve_dense(
                        lc, cache["chunks"], cache["pages"], cache["embeddings"], query, top_k
                    )
            else:  # bm25
                if cache["bm25"] is None:
                    scored = []
                else:
                    bm25_results = cache["bm25"].retrieve(query, top_k)
                    # BM25 returns (chunk, page, score)
                    scored = [(chunk, page, score) for chunk, page, score in bm25_results]
        except Exception as e:
            print(f"    [ERROR] query id={item['id']}: {type(e).__name__}: {e}")
            traceback.print_exc()
            scored = []
        retrieval_time = round(time.perf_counter() - t0, 3)

        retrieved_chunks = [c for c, _p, _s in scored]
        retrieved_pages = [p for _c, p, _s in scored]

        # ---- Semantic relevance (FAIR VERSION) ----
        # ALWAYS embed the reference answer and each chunk in isolation.
        # This ensures all methods are judged on the same footing.
        if reference_answer and retrieved_chunks:
            answer_emb = embed_text(lc, reference_answer)
            sims = []
            for chunk in retrieved_chunks:
                chunk_emb = embed_text(lc, chunk)
                sim = F.cosine_similarity(answer_emb.unsqueeze(0), chunk_emb.unsqueeze(0)).item()
                sims.append(sim)
            relevant = [sim >= similarity_threshold for sim in sims]
        else:
            relevant = [False] * len(retrieved_chunks)

        # Compute metrics
        rec = 1.0 if any(relevant) else 0.0
        prec = sum(relevant) / len(relevant) if relevant else 0.0
        mrr_val = 0.0
        for rank, rel in enumerate(relevant, start=1):
            if rel:
                mrr_val = 1.0 / rank
                break

        # Rank of correct page (location metric, unchanged)
        rank = None
        for i, page in enumerate(retrieved_pages, start=1):
            p = page.get("page_start") if isinstance(page, dict) else page
            p_end = page.get("page_end") if isinstance(page, dict) else page
            if correct_page is not None and (p == correct_page or (
                isinstance(p, int) and isinstance(p_end, int) and p <= correct_page <= p_end
            )):
                rank = i
                break

        avg_sim = round(sum(sims) / len(sims), 4) if sims else 0.0

        row = {
            "method": method,
            "id": item["id"],
            "document": item["document"],
            "query": query,
            "correct_page": correct_page,
            "similarity_threshold": similarity_threshold,
            "top_k": top_k,
            "retrieved_pages": [
                (p.get("page_start") if isinstance(p, dict) else p) for p in retrieved_pages
            ],
            "retrieval_time_s": retrieval_time,
            "recall": rec,
            "precision": prec,
            "mrr": mrr_val,
            "rank_of_correct_page": rank,
            "avg_similarity": avg_sim,
            "_top_chunk_preview": retrieved_chunks[0][:120] if retrieved_chunks else "",
        }
        results.append(row)

        hit_str = f"rank={rank}" if rank else "MISS"
        print(f"    [Q{item['id']:03d}] recall={rec:.0f}  prec={prec:.2f}  "
              f"mrr={mrr_val:.3f}  {hit_str}  sim={avg_sim:.3f}  time={retrieval_time}s")

    # Aggregate statistics
    n = len(results)

    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    rank_hits = [r["rank_of_correct_page"] for r in results if r["rank_of_correct_page"] is not None]
    recall_avg = avg("recall")
    prec_avg = avg("precision")

    aggregate = {
        "method": method,
        "n_queries": n,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "recall_pct": round(recall_avg * 100, 1) if recall_avg is not None else None,
        "precision_pct": round(prec_avg * 100, 1) if prec_avg is not None else None,
        "mrr": avg("mrr"),
        "avg_retrieval_time_s": avg("retrieval_time_s"),
        "correct_page_found_pct": round(len(rank_hits) / n * 100, 1) if n else 0,
    }

    return results, aggregate


def main():
    parser = argparse.ArgumentParser(
        description="Unified retrieval eval using semantic similarity against reference_answer."
    )
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--pdf_dir", required=True)
    parser.add_argument("--windowed_dir", default=config.PERSIST_DIR)
    parser.add_argument("--naive_dir", default=None)
    parser.add_argument("--out", default="evaluation/retrieval_results_v2")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                        help="Threshold for considering a retrieved chunk relevant (cosine similarity).")
    args = parser.parse_args()

    top_k = args.top_k or config.TOP_K

    eval_set_path = _resolve_dir(args.eval_set)
    pdf_dir = _resolve_dir(args.pdf_dir)
    windowed_dir = _resolve_dir(args.windowed_dir)
    naive_dir = _resolve_dir(args.naive_dir) if args.naive_dir else (windowed_dir.parent / "chroma_db_naive")
    out_dir = _resolve_dir(args.out)

    with open(eval_set_path) as f:
        eval_set = json.load(f)
    if args.limit:
        eval_set = eval_set[:args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[init] Loading model and tokenizer...")
    lc = LateChunking(model_name=config.MODEL, tokenizer_name=config.TOKENIZER)
    vs_windowed = VectorStore(persist_dir=str(windowed_dir))
    vs_naive = VectorStore(persist_dir=str(naive_dir))
    print(f"[init] project root:   {PROJECT_ROOT}")
    print(f"[init] eval set:       {eval_set_path}")
    print(f"[init] pdf dir:        {pdf_dir}")
    print(f"[init] windowed store: {windowed_dir}")
    print(f"[init] naive store:    {naive_dir}")
    print(f"[init] similarity threshold: {args.similarity_threshold}")
    print(f"[init] Ready. Running {len(eval_set)} queries x {len(args.methods)} method(s), top_k={top_k}\n")
    print("─" * 70)

    all_results = []
    all_aggregates = {}

    for method in args.methods:
        print(f"\n{'='*70}\n  METHOD: {method}\n{'='*70}")
        results, aggregate = run_eval(
            method, eval_set, pdf_dir, lc, vs_windowed, vs_naive,
            top_k, similarity_threshold=args.similarity_threshold
        )
        all_results.extend(results)
        all_aggregates[method] = aggregate

        method_out = out_dir / f"retrieval_{method}.json"
        with open(method_out, "w") as f:
            json.dump({
                "meta": {
                    "run_at": datetime.now().isoformat(),
                    "method": method,
                    "eval_set": args.eval_set,
                    "top_k": top_k,
                    "similarity_threshold": args.similarity_threshold,
                    "n_run": len(results),
                },
                "aggregate": aggregate,
                "results": results,
            }, f, indent=2)
        print(f"\n  -> saved {method_out}")

    csv_path = out_dir / "retrieval_all_methods.csv"
    fields = ["method", "id", "document", "query", "correct_page", "top_k",
              "similarity_threshold", "retrieval_time_s", "recall", "precision", "mrr",
              "rank_of_correct_page", "avg_similarity"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k) for k in fields})

    agg_path = out_dir / "aggregate_comparison.csv"
    agg_fields = ["method", "n_queries", "top_k", "similarity_threshold",
                  "recall_pct", "precision_pct", "mrr",
                  "avg_retrieval_time_s", "correct_page_found_pct"]
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for method, agg in all_aggregates.items():
            writer.writerow({k: agg.get(k) for k in agg_fields})

    w = 70
    print(f"\n{'═'*w}")
    print(f"  METHOD COMPARISON")
    print(f"{'─'*w}")
    print(f"  {'method':<10} {'recall%':>8} {'prec%':>7} {'mrr':>6} {'time(s)':>8} {'pagehit%':>9}")
    for method, agg in all_aggregates.items():
        print(f"  {method:<10} {agg['recall_pct']!s:>8} {agg['precision_pct']!s:>7} "
              f"{agg['mrr']!s:>6} {agg['avg_retrieval_time_s']!s:>8} {agg['correct_page_found_pct']!s:>9}")
    print(f"{'═'*w}")
    print(f"\n  Saved:")
    print(f"    {csv_path}")
    print(f"    {agg_path}")
    print(f"\n  avg_similarity is cosine similarity between chunk and reference_answer.")
    print(f"  The threshold used was {args.similarity_threshold}.")
    print(f"  **All methods are judged using the same isolated embedding space.**\n")


if __name__ == "__main__":
    main()