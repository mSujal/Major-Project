"""
Can a weak LLM with chain of thoughts reflect a cognitive load experienced by a student taking a test?
"""

import re
import os
import time
import json
import argparse
from groq import Groq
import numpy as np
from collections import defaultdict, Counter
from dotenv import load_dotenv
from pathlib import Path

from feature_extraction import extract_features

load_dotenv(Path(__file__).parent / ".env")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set")

MODEL = "openai/gpt-oss-20b"

def load_questions_from_files(file_paths):
    all_q = []
    for fp in file_paths:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            all_q.extend(data)
        elif isinstance(data, dict):
            if "questions" in data:
                all_q.extend(data["questions"])
            elif "question" in data and "options" in data:
                all_q.append(data)
            else:
                print(f"Warning: Unrecognized structure in {fp}. Skipping.")
        else:
            print(f"Warning: Unexpected type in {fp}. Skipping.")
    return all_q

def solve_with_reasoning(q, client):
    opts = "\n".join([f"{k}) {v}" for k, v in q["options"].items()])
    system = (
        "You are a diligent student taking a multiple-choice test. "
        "Reason aloud step-by-step before giving your final answer. "
        "Do not skip reasoning steps."
    )
    prompt = f"""
    Question: {q["question"]}
    {opts}

    Follow these steps:
    1. Restate the question.
    2. Analyze each option (A-D) – explain why correct or incorrect.
    3. Eliminate incorrect options.
    4. Choose the best answer and explain.
    5. On the last line, output your FINAL ANSWER as a single letter.

    Begin:
    """
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1000
    )
    end = time.perf_counter()
    raw = resp.choices[0].message.content
    matches = re.findall(r'\b([A-D])\b', raw)
    pred = matches[-1] if matches else "X"
    return {
        "raw_output": raw,
        "latency": end - start,
        "tokens": resp.usage.completion_tokens,
        "completion_time": resp.usage.completion_time,
        "predicted": pred,
        "correct": pred == q["correct_answer"]
    }

def normalize(vals):
    mn, mx = min(vals), max(vals)
    if mx - mn == 0:
        return [0.5] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--show-reasoning", nargs="?", const="all", default=None,
                        help="Show full reasoning. Use --show-reasoning N to show first N questions, or --show-reasoning all for all.")
    args = parser.parse_args()


    questions = load_questions_from_files(args.files)
    if not questions:
        print("No questions loaded.")
        return
    print(f"Loaded {len(questions)} questions.")

    # Determine how many reasoning outputs to show
    show_limit = None
    if args.show_reasoning is not None:
        if args.show_reasoning.lower() == "all":
            show_limit = len(questions)
        else:
            try:
                show_limit = int(args.show_reasoning)
            except ValueError:
                show_limit = len(questions)

    client = Groq(api_key=api_key)
    results_by_diff = defaultdict(list)
    all_results = []   # (idx, q, result)

    for i, q in enumerate(questions, 1):
        diff = q.get("difficulty", "Unknown")
        print(f"[{i}/{len(questions)}] {diff}: {q['question'][:40]}...")
        res = solve_with_reasoning(q, client)
        results_by_diff[diff].append(res)
        all_results.append((i-1, q, res))

        # Show reasoning if requested
        if show_limit is not None and i <= show_limit:
            print("\n" + "="*80)
            print(f"REASONING FOR QUESTION #{i}")
            print("="*80)
            print(res["raw_output"])
            print("="*80)
            print(f"Predicted: {res['predicted']} | Correct: {res['correct']}\n")

        print(f"""
            Question: {q}
            {'='*60}
            Latency: {res['latency']:.2f}s
            Completion: {res['completion_time']:.2f}s
            Tokens: {res['tokens']}
            Predicted: {res['predicted']}
            Correct: {res['correct']}
        """)

    print("\nExtracting features...")
    feature_list = [extract_features(q) for _, q, _ in all_results]

    print("Computing empirical difficulty...")
    comp_times = [r["completion_time"] for _, _, r in all_results]
    tokens = [r["tokens"] for _, _, r in all_results]
    corrects = [1 if r["correct"] else 0 for _, _, r in all_results]

    norm_comp = normalize(comp_times)
    norm_tok = normalize(tokens)
    norm_inc = [1 - c for c in normalize(corrects)]

    f_names = list(feature_list[0].keys())
    f_vals = {name: [f[name] for f in feature_list] for name in f_names}
    norm_f = {name: normalize(vals) for name, vals in f_vals.items()}

    weights = {
        "completion_time": 0.25,
        "tokens": 0.15,
        "incorrect_penalty": 0.10,
        "gunning_fog": 0.04,
        "flesch_kincaid_grade": 0.03,
        "stem_word_count": 0.03,
        "avg_distractor_overlap": 0.06,
        "max_distractor_similarity": 0.06,
        "negation_presence": 0.04,
        "clause_count": 0.03,
        "parse_tree_depth": 0.03,
        "quantitative_flag": 0.03,
        "content_word_density": 0.02,
        "distractor_count": 0.02,
        "avg_distractor_similarity": 0.02,
        "distractor_length_variance": 0.02,
        "modal_presence": 0.02,
        "noun_ratio": 0.01,
        "verb_ratio": 0.01,
        "adj_ratio": 0.01,
        "stem_sentence_count": 0.01,
        "avg_word_length": 0.01,
    }

    raw_scores = []
    for i in range(len(all_results)):
        score = 0.0
        score += weights.get("completion_time", 0) * norm_comp[i]
        score += weights.get("tokens", 0) * norm_tok[i]
        score += weights.get("incorrect_penalty", 0) * norm_inc[i]
        for name, w in weights.items():
            if name in norm_f:
                score += w * norm_f[name][i]
        raw_scores.append(score)

    mn, mx = min(raw_scores), max(raw_scores)
    final_scores = [(s - mn) / (mx - mn) if mx != mn else 0.5 for s in raw_scores]

    labels = []
    for s in final_scores:
        if s < 0.33:
            labels.append("Easy")
        elif s < 0.66:
            labels.append("Medium")
        else:
            labels.append("Hard")

    for i, (idx, q, r) in enumerate(all_results):
        r["empirical_difficulty"] = labels[i]
        contribs = []
        for name, w in weights.items():
            if name in norm_f:
                val = norm_f[name][i]
                contribs.append((name, val * w))
            elif name == "completion_time":
                contribs.append((name, norm_comp[i] * w))
            elif name == "tokens":
                contribs.append((name, norm_tok[i] * w))
            elif name == "incorrect_penalty":
                contribs.append((name, norm_inc[i] * w))
        contribs.sort(key=lambda x: x[1], reverse=True)
        top3 = [f"{c[0]}={c[1]:.2f}" for c in contribs[:3]]
        r["justification"] = (
            f"Empirical: {labels[i]}. Score: {final_scores[i]:.3f}. "
            f"Comp: {r['completion_time']:.2f}s, tokens: {r['tokens']}, "
            f"{'correct' if r['correct'] else 'incorrect'}, Bloom: {q.get('bloom_level', 'Unknown')}. "
            f"Top: {', '.join(top3)}"
        )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY BY ORIGINAL DIFFICULTY")
    print("="*60)
    summary = []
    for level in ["Easy", "Medium", "Hard"]:
        data = results_by_diff.get(level, [])
        if not data:
            print(f"{level}: No data")
            continue
        avg_comp = np.mean([d["completion_time"] for d in data])
        avg_tok = np.mean([d["tokens"] for d in data])
        avg_lat = np.mean([d["latency"] for d in data])
        acc = np.mean([d["correct"] for d in data]) * 100
        summary.append((level, avg_comp, avg_tok, acc))
        print(f"{level}:")
        print(f"  Comp: {avg_comp:.2f}s, Tokens: {avg_tok:.1f}, Latency: {avg_lat:.2f}s, Acc: {acc:.1f}%")

    # Cross-tab with Bloom
    bloom_map = {
        "Remember": "Easy",
        "Understand": "Easy",
        "Apply": "Medium",
        "Analyze": "Medium",
        "Evaluate": "Hard",
        "Create": "Hard"
    }
    cross = defaultdict(Counter)
    for i, (idx, q, r) in enumerate(all_results):
        b = q.get("bloom_level", "Unknown")
        bm = bloom_map.get(b, "Unknown")
        cross[bm][r["empirical_difficulty"]] += 1

    print("\n" + "="*60)
    print("EMPIRICAL vs BLOOM CROSS-TAB")
    print("="*60)
    print("{:<12} {:<12} {:<12} {:<12} {:<12}".format("Bloom\\Emp", "Easy", "Medium", "Hard", "Total"))
    print("-"*60)
    for b in ["Easy", "Medium", "Hard"]:
        row = cross.get(b, Counter())
        total = sum(row.values())
        print("{:<12} {:<12} {:<12} {:<12} {:<12}".format(
            b, row.get("Easy", 0), row.get("Medium", 0), row.get("Hard", 0), total
        ))

    if args.output:
        out = []
        for i, (idx, q, r) in enumerate(all_results):
            out.append({
                "question": q["question"],
                "options": q["options"],
                "correct_answer": q["correct_answer"],
                "bloom_level": q.get("bloom_level", "Unknown"),
                "empirical_difficulty": r["empirical_difficulty"],
                "justification": r["justification"],
                "raw_reasoning": r["raw_output"],   # <-- now saved
                "completion_time": r["completion_time"],
                "tokens": r["tokens"],
                "correct": r["correct"],
                "predicted": r["predicted"],
                "latency": r["latency"],
                "features": feature_list[i]
            })
        with open(args.output, 'w') as f:
            json.dump({
                "summary": summary,
                "cross_tabulation": {k: dict(v) for k, v in cross.items()},
                "detailed_results": out
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()