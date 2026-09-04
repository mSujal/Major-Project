import json
import argparse
import csv
import math
from pathlib import Path
import numpy as np

# ---- Optional Libraries ----
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

try:
    import textstat
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    print("Warning: textstat not installed. Readability scores will be skipped. Install with: pip install textstat")

# ---------- Bloom's Taxonomy Keyword Mapping ----------
BLOOM_VERBS = {
    "remember": [
        "define", "identify", "list", "name", "recall", "recognize", "state", "memorize",
        "repeat", "reproduce", "select", "label", "match", "cite", "outline", "quote"
    ],
    "understand": [
        "describe", "explain", "paraphrase", "summarize", "interpret", "classify", "compare",
        "contrast", "discuss", "distinguish", "estimate", "infer", "predict", "restate",
        "translate", "give examples", "rephrase", "exemplify", "illustrate"
    ],
    "apply": [
        "apply", "demonstrate", "calculate", "complete", "illustrate", "show", "solve",
        "modify", "operate", "practice", "schedule", "sketch", "use", "compute", "implement",
        "execute", "perform", "produce", "relate", "transfer"
    ],
    "analyze": [
        "analyze", "categorize", "classify", "compare", "contrast", "differentiate", "distinguish",
        "examine", "investigate", "organize", "outline", "test", "break down", "diagram",
        "dissect", "inspect", "question", "separate", "structure", "appraise"
    ],
    "evaluate": [
        "evaluate", "argue", "assess", "defend", "judge", "justify", "support", "validate",
        "critique", "criticize", "debate", "determine", "estimate", "interpret", "measure",
        "rank", "rate", "recommend", "select", "verify"
    ],
    "create": [
        "create", "compose", "construct", "design", "develop", "formulate", "hypothesize",
        "invent", "produce", "write", "assemble", "build", "combine", "compile", "generate",
        "integrate", "modify", "plan", "propose", "reconstruct"
    ]
}

def get_bloom_level(text, provided=None):
    if provided and provided.lower() in BLOOM_VERBS:
        return provided.lower()
    if not isinstance(text, str):
        return "unknown"
    text_lower = text.lower()
    for level, verbs in BLOOM_VERBS.items():
        for verb in verbs:
            if verb in text_lower:
                return level
    return "unknown"

# ---------- Tokenization & Similarity Helpers ----------
def tokenize(text):
    if not isinstance(text, str):
        return set()
    text = text.lower().strip()
    for p in ['.', ',', '?', '!', ';', ':', '"', "'", '(', ')', '-', '_']:
        text = text.replace(p, ' ')
    return set(text.split())

def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / union if union > 0 else 0.0

def load_mcqs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ['questions', 'mcqs', 'data']:
            if key in data and isinstance(data[key], list):
                return data[key]
        if 'question' in data:
            return [data]
    raise ValueError("Could not parse JSON: expected a list or an object with 'questions' or 'mcqs' key.")

def parse_options(options_obj):
    if isinstance(options_obj, list):
        return options_obj
    if isinstance(options_obj, dict):
        return [options_obj[k] for k in sorted(options_obj.keys()) if k in options_obj]
    return []

def correct_letter_to_index(letter, num_options):
    if isinstance(letter, int):
        if 0 <= letter < num_options:
            return letter
        return -1
    if isinstance(letter, str):
        letter = letter.strip().upper()
        if len(letter) == 1 and letter.isalpha():
            idx = ord(letter) - ord('A')
            if 0 <= idx < num_options:
                return idx
        if letter.isdigit():
            idx = int(letter)
            if 0 <= idx < num_options:
                return idx
    return -1

# ---------- Main Feature Extraction ----------
def extract_features(mcq, embedder=None):
    question = mcq.get('question', '')
    options_raw = mcq.get('options', [])
    options = parse_options(options_raw)
    correct_ans = mcq.get('correct_answer', -1)
    correct_idx = correct_letter_to_index(correct_ans, len(options))
    if correct_idx == -1:
        correct_idx = 0  # fallback

    explanation = mcq.get('explanation', '')
    bloom_provided = mcq.get('bloom_level', '')
    citation = mcq.get('citation', '')

    # ---- Basic Tokenization ----
    q_tokens = tokenize(question)
    expl_tokens = tokenize(explanation)
    tokenized_opts = [tokenize(opt) for opt in options]

    correct_text = options[correct_idx] if 0 <= correct_idx < len(options) else ""
    correct_tokens = tokenize(correct_text)

    # ---- Stem ----
    stem_word_count = len(q_tokens)
    stem_char_count = len(question.strip())
    stem_unique_ratio = len(q_tokens) / stem_word_count if stem_word_count > 0 else 0

    # ---- Options ----
    num_options = len(options)
    option_word_counts = [len(tok) for tok in tokenized_opts]
    option_char_counts = [len(opt.strip()) for opt in options]
    avg_option_word_count = sum(option_word_counts) / num_options if num_options > 0 else 0
    std_option_word_count = math.sqrt(sum((x - avg_option_word_count)**2 for x in option_word_counts) / num_options) if num_options > 0 else 0
    avg_option_char_count = sum(option_char_counts) / num_options if num_options > 0 else 0

    # ---- Option Similarity ----
    pairwise_sims = []
    for i in range(num_options):
        for j in range(i+1, num_options):
            pairwise_sims.append(jaccard_similarity(tokenized_opts[i], tokenized_opts[j]))
    pairwise_avg_jaccard = sum(pairwise_sims) / len(pairwise_sims) if pairwise_sims else 0.0

    distractor_sims_to_correct = []
    if 0 <= correct_idx < num_options:
        correct_toks = tokenized_opts[correct_idx]
        for i, opt_toks in enumerate(tokenized_opts):
            if i != correct_idx:
                distractor_sims_to_correct.append(jaccard_similarity(correct_toks, opt_toks))
    avg_distractor_sim = sum(distractor_sims_to_correct) / len(distractor_sims_to_correct) if distractor_sims_to_correct else 0.0
    min_distractor_sim = min(distractor_sims_to_correct) if distractor_sims_to_correct else 0.0
    max_distractor_sim = max(distractor_sims_to_correct) if distractor_sims_to_correct else 0.0

    # ---- Explanation Correspondence ----
    expl_to_question_jaccard = jaccard_similarity(expl_tokens, q_tokens)
    expl_to_correct_jaccard = jaccard_similarity(expl_tokens, correct_tokens)

    distractor_tokens = [tok for i, tok in enumerate(tokenized_opts) if i != correct_idx]
    distractor_sims_expl = [jaccard_similarity(expl_tokens, d) for d in distractor_tokens]
    expl_to_distractor_avg_jaccard = sum(distractor_sims_expl) / len(distractor_sims_expl) if distractor_sims_expl else 0.0

    combined_q_c = q_tokens.union(correct_tokens)
    new_tokens = expl_tokens - combined_q_c
    expl_new_info_ratio = len(new_tokens) / len(expl_tokens) if expl_tokens else 0.0

    expl_contains_correct_verbatim = correct_text.lower() in explanation.lower() if correct_text else False
    explanation_word_count = len(expl_tokens)

    # ---- Readability ----
    flesch_reading_ease = None
    flesch_kincaid_grade = None
    smog_index = None
    coleman_liau_index = None
    if READABILITY_AVAILABLE and question:
        try:
            flesch_reading_ease = round(textstat.flesch_reading_ease(question), 2)
            flesch_kincaid_grade = round(textstat.flesch_kincaid_grade(question), 2)
            smog_index = round(textstat.smog_index(question), 2)
            coleman_liau_index = round(textstat.coleman_liau_index(question), 2)
        except:
            pass

    # ---- Bloom ----
    bloom_level = get_bloom_level(question, bloom_provided)

    # ---- Semantic Similarity (optional) ----
    expl_semantic_to_question_cos = None
    expl_semantic_to_correct_cos = None
    if embedder is not None:
        if question and explanation:
            emb_q = embedder.encode(question)
            emb_expl = embedder.encode(explanation)
            cos_sim = np.dot(emb_q, emb_expl) / (np.linalg.norm(emb_q) * np.linalg.norm(emb_expl))
            expl_semantic_to_question_cos = round(float(cos_sim), 4)
        if correct_text and explanation:
            emb_c = embedder.encode(correct_text)
            emb_expl = embedder.encode(explanation)
            cos_sim = np.dot(emb_c, emb_expl) / (np.linalg.norm(emb_c) * np.linalg.norm(emb_expl))
            expl_semantic_to_correct_cos = round(float(cos_sim), 4)

    # ---- Build Result Dictionary ----
    result = {
        'stem_word_count': stem_word_count,
        'stem_char_count': stem_char_count,
        'stem_unique_ratio': round(stem_unique_ratio, 3),
        'num_options': num_options,
        'avg_option_word_count': round(avg_option_word_count, 2),
        'std_option_word_count': round(std_option_word_count, 2),
        'avg_option_char_count': round(avg_option_char_count, 2),
        'correct_answer_index': correct_idx,
        'pairwise_avg_jaccard': round(pairwise_avg_jaccard, 3),
        'avg_distractor_jaccard_to_correct': round(avg_distractor_sim, 3),
        'min_distractor_jaccard_to_correct': round(min_distractor_sim, 3),
        'max_distractor_jaccard_to_correct': round(max_distractor_sim, 3),
        'explanation_word_count': explanation_word_count,
        'expl_to_question_jaccard': round(expl_to_question_jaccard, 3),
        'expl_to_correct_jaccard': round(expl_to_correct_jaccard, 3),
        'expl_to_distractor_avg_jaccard': round(expl_to_distractor_avg_jaccard, 3),
        'expl_new_info_ratio': round(expl_new_info_ratio, 3),
        'expl_contains_correct_verbatim': expl_contains_correct_verbatim,
        'citation_length': len(citation.strip()),
        'bloom_taxonomy_level': bloom_level,
        'flesch_reading_ease': flesch_reading_ease,
        'flesch_kincaid_grade': flesch_kincaid_grade,
        'smog_index': smog_index,
        'coleman_liau_index': coleman_liau_index,
    }
    if expl_semantic_to_question_cos is not None:
        result['expl_semantic_to_question_cos'] = expl_semantic_to_question_cos
        result['expl_semantic_to_correct_cos'] = expl_semantic_to_correct_cos

    return result

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description='Extract comprehensive features from generated MCQs.')
    parser.add_argument('--input', required=True, help='JSON file with generated MCQs')
    parser.add_argument('--output', default='mcq_features.csv', help='Output CSV for per-question features')
    parser.add_argument('--subject_field', default='subject', help='Field name for subject (if any)')
    parser.add_argument('--subject_override', default=None,
                        help='Override subject for all MCQs (ignores --subject_field)')
    parser.add_argument('--aggregate', default='aggregate_stats.csv', help='Output CSV for aggregated stats')
    parser.add_argument('--use_embeddings', action='store_true', help='Use sentence-transformers for semantic similarity')
    args = parser.parse_args()

    embedder = None
    if args.use_embeddings:
        if not SEMANTIC_AVAILABLE:
            print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
            return
        print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")

    mcqs = load_mcqs(args.input)
    print(f"Loaded {len(mcqs)} MCQs.")
    if mcqs:
        print("First MCQ keys:", list(mcqs[0].keys()))

    feature_list = []
    for mcq in mcqs:
        features = extract_features(mcq, embedder)
        if args.subject_override:
            subject = args.subject_override
        else:
            subject = mcq.get(args.subject_field, 'Overall')
        features['subject'] = subject if subject else 'Overall'
        feature_list.append(features)

    # Define columns for export
    columns = [
        'subject', 'bloom_taxonomy_level',
        'stem_word_count', 'stem_char_count', 'stem_unique_ratio',
        'num_options',
        'avg_option_word_count', 'std_option_word_count', 'avg_option_char_count',
        'correct_answer_index',
        'pairwise_avg_jaccard',
        'avg_distractor_jaccard_to_correct',
        'min_distractor_jaccard_to_correct',
        'max_distractor_jaccard_to_correct',
        'explanation_word_count',
        'expl_to_question_jaccard',
        'expl_to_correct_jaccard',
        'expl_to_distractor_avg_jaccard',
        'expl_new_info_ratio',
        'expl_contains_correct_verbatim',
        'citation_length',
        'flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index', 'coleman_liau_index'
    ]
    if args.use_embeddings and feature_list and 'expl_semantic_to_question_cos' in feature_list[0]:
        columns += ['expl_semantic_to_question_cos', 'expl_semantic_to_correct_cos']

    # ---- Write per-question CSV ----
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in feature_list:
            writer.writerow({k: row.get(k, '') for k in columns})
    print(f"Per-question features saved to {args.output}")

    # ---- Aggregate statistics ----
    subjects = set(row['subject'] for row in feature_list)
    aggregates = []
    numeric_fields = [c for c in columns if c not in ['subject', 'bloom_taxonomy_level', 'correct_answer_index', 'expl_contains_correct_verbatim']]

    for sub in subjects:
        sub_rows = [r for r in feature_list if r['subject'] == sub]
        if not sub_rows:
            continue
        agg = {'subject': sub, 'n': len(sub_rows)}
        for field in numeric_fields:
            vals = [r[field] for r in sub_rows if r.get(field) is not None and r.get(field) != '']
            if vals:
                agg[f'avg_{field}'] = round(sum(vals) / len(vals), 3)
                if len(vals) > 1:
                    agg[f'std_{field}'] = round(math.sqrt(sum((x - agg[f'avg_{field}'])**2 for x in vals) / len(vals)), 3)
                else:
                    agg[f'std_{field}'] = 0.0
            else:
                agg[f'avg_{field}'] = 0.0
                agg[f'std_{field}'] = 0.0
        # Correct answer distribution
        indices = [r['correct_answer_index'] for r in sub_rows if r.get('correct_answer_index') != -1]
        if indices:
            for i in range(4):
                agg[f'pct_ans_{i}'] = round(sum(1 for idx in indices if idx == i) / len(indices) * 100, 1)
        # Bloom distribution
        blooms = [r['bloom_taxonomy_level'] for r in sub_rows if r.get('bloom_taxonomy_level') != 'unknown']
        if blooms:
            for level in ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']:
                agg[f'pct_bloom_{level}'] = round(sum(1 for b in blooms if b == level) / len(blooms) * 100, 1)
        # Verbatim copy
        verbatim_vals = [r['expl_contains_correct_verbatim'] for r in sub_rows if 'expl_contains_correct_verbatim' in r]
        if verbatim_vals:
            agg['pct_expl_verbatim'] = round(sum(1 for v in verbatim_vals if v) / len(verbatim_vals) * 100, 1)
        aggregates.append(agg)

    if aggregates:
        sample = aggregates[0]
        agg_columns = ['subject', 'n']
        for key in sorted(sample.keys()):
            if key not in ['subject', 'n']:
                agg_columns.append(key)
    else:
        agg_columns = ['subject', 'n']

    with open(args.aggregate, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=agg_columns)
        writer.writeheader()
        for row in aggregates:
            writer.writerow({k: row.get(k, '') for k in agg_columns})
    print(f"Aggregated stats saved to {args.aggregate}")

    # ---- Console Summary ----
    overall = [r for r in feature_list if r['subject'] == 'Overall']
    if overall:
        print("\n=== OVERALL SUMMARY ===")
        print(f"Total MCQs: {len(overall)}")
        print(f"Avg Stem Words: {sum(r['stem_word_count'] for r in overall)/len(overall):.1f}")
        print(f"Avg Option Words: {sum(r['avg_option_word_count'] for r in overall)/len(overall):.2f}")
        print(f"Avg Pairwise Jaccard: {sum(r['pairwise_avg_jaccard'] for r in overall)/len(overall):.3f}")
        print(f"Avg Distractor Similarity to Correct: {sum(r['avg_distractor_jaccard_to_correct'] for r in overall)/len(overall):.3f}")
        print(f"Avg Explanation Words: {sum(r['explanation_word_count'] for r in overall)/len(overall):.1f}")
        print(f"Expl -> Question Jaccard: {sum(r['expl_to_question_jaccard'] for r in overall)/len(overall):.3f}")
        print(f"Expl -> Correct Jaccard: {sum(r['expl_to_correct_jaccard'] for r in overall)/len(overall):.3f}")
        print(f"New Information Ratio: {sum(r['expl_new_info_ratio'] for r in overall)/len(overall):.3f}")
        print(f"Verbatim Copy: {sum(1 for r in overall if r['expl_contains_correct_verbatim'])/len(overall)*100:.1f}%")
        if READABILITY_AVAILABLE:
            print(f"Avg Flesch Reading Ease: {sum(r['flesch_reading_ease'] for r in overall if r['flesch_reading_ease'])/len(overall):.1f}")
            print(f"Avg Flesch-Kincaid Grade: {sum(r['flesch_kincaid_grade'] for r in overall if r['flesch_kincaid_grade'])/len(overall):.1f}")
        blooms = [r['bloom_taxonomy_level'] for r in overall if r.get('bloom_taxonomy_level') != 'unknown']
        if blooms:
            print("Bloom Distribution:")
            for level in ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']:
                pct = sum(1 for b in blooms if b == level) / len(blooms) * 100
                print(f"  {level}: {pct:.1f}%")

    print("\nDone! Use the CSVs for plots and tables in Section 4.2.")

if __name__ == '__main__':
    main()