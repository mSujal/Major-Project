import numpy as np
from collections import Counter

try:
    import textstat
except ImportError:
    textstat = None

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    for resource in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
    _NLTK = True
    STOPWORDS = set(stopwords.words('english'))
except (ImportError, LookupError) as e:
    _NLTK = False
    STOPWORDS = set()
    print(f"Warning: NLTK not fully available: {e}")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    _SPACY = True
except (ImportError, OSError):
    _SPACY = False
    nlp = None

try:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    _EMBED = True
except ImportError:
    _EMBED = False
    embed_model = None


def extract_features(q):
    stem = q["question"]
    options = list(q["options"].values())
    correct_answer = q["correct_answer"]
    correct_text = q["options"][correct_answer]
    distractors = [opt for k, opt in q["options"].items() if k != correct_answer]

    features = {}
    full_text = stem + " " + " ".join(options)

    # Readability
    if textstat:
        features["flesch_reading_ease"] = textstat.flesch_reading_ease(full_text)
        features["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(full_text)
        features["ari"] = textstat.automated_readability_index(full_text)
        features["gunning_fog"] = textstat.gunning_fog(full_text)
        features["coleman_liau"] = textstat.coleman_liau_index(full_text)
        features["smog"] = textstat.smog_index(full_text)
    else:
        for f in ["flesch_reading_ease", "flesch_kincaid_grade", "ari", "gunning_fog", "coleman_liau", "smog"]:
            features[f] = 0.0

    # Tokenization
    if _NLTK:
        stem_words = word_tokenize(stem)
        correct_words = word_tokenize(correct_text)
        distractor_words = [word_tokenize(d) for d in distractors]
        all_words = word_tokenize(full_text)
    else:
        stem_words = stem.split()
        correct_words = correct_text.split()
        distractor_words = [d.split() for d in distractors]
        all_words = full_text.split()

    features["stem_word_count"] = len(stem_words)
    features["correct_word_count"] = len(correct_words)
    features["avg_distractor_word_count"] = np.mean([len(w) for w in distractor_words]) if distractors else 0.0

    if _NLTK:
        features["stem_sentence_count"] = len(sent_tokenize(stem))
        features["correct_sentence_count"] = len(sent_tokenize(correct_text))
        features["avg_distractor_sentence_count"] = np.mean([len(sent_tokenize(d)) for d in distractors]) if distractors else 0.0
    else:
        features["stem_sentence_count"] = stem.count('.') + stem.count('!') + stem.count('?')
        features["correct_sentence_count"] = correct_text.count('.') + correct_text.count('!') + correct_text.count('?')
        features["avg_distractor_sentence_count"] = np.mean([d.count('.') + d.count('!') + d.count('?') for d in distractors]) if distractors else 0.0

    features["avg_word_length"] = np.mean([len(w) for w in all_words]) if all_words else 0.0

    avg_option_len = (len(correct_words) + sum(len(w) for w in distractor_words)) / (1 + len(distractors)) if distractors else len(correct_words)
    features["stem_to_option_ratio"] = len(stem_words) / max(avg_option_len, 1.0)

    # Syntactic
    if _SPACY and nlp:
        doc = nlp(stem)
        def max_depth(token):
            if not list(token.children):
                return 1
            return 1 + max(max_depth(child) for child in token.children)
        features["parse_tree_depth"] = max(max_depth(sent.root) for sent in doc.sents) if doc.sents else 0

        pos_counts = Counter(token.pos_ for token in doc)
        total = len(doc)
        features["noun_ratio"] = pos_counts.get("NOUN", 0) / total if total else 0.0
        features["verb_ratio"] = pos_counts.get("VERB", 0) / total if total else 0.0
        features["adj_ratio"] = pos_counts.get("ADJ", 0) / total if total else 0.0
    else:
        features["parse_tree_depth"] = 0
        features["noun_ratio"] = 0.0
        features["verb_ratio"] = 0.0
        features["adj_ratio"] = 0.0

    subord = {"if", "because", "although", "while", "when", "unless", "since", "whereas", "wherever"}
    features["clause_count"] = sum(1 for w in stem_words if w.lower() in subord) + 1

    # Distractor features
    features["distractor_count"] = len(distractors)

    correct_set = set(correct_words)
    overlap_scores = []
    for d_words in distractor_words:
        d_set = set(d_words)
        if not correct_set or not d_set:
            overlap_scores.append(0.0)
        else:
            overlap = len(correct_set.intersection(d_set)) / len(correct_set.union(d_set))
            overlap_scores.append(overlap)
    features["avg_distractor_overlap"] = np.mean(overlap_scores) if overlap_scores else 0.0

    distractor_lengths = [len(w) for w in distractor_words]
    features["distractor_length_variance"] = np.var(distractor_lengths) if distractors else 0.0

    # Lexical sophistication
    if STOPWORDS:
        content_words = [w for w in all_words if w.lower() not in STOPWORDS and w.isalpha()]
    else:
        content_words = [w for w in all_words if w.isalpha()]
    features["content_word_density"] = len(content_words) / max(len(all_words), 1)

    # Complexity markers
    neg_words = {"not", "no", "never", "none", "neither", "nor", "except", "without"}
    features["negation_presence"] = int(any(w.lower() in neg_words for w in stem_words))
    modals = {"can", "could", "may", "might", "would", "should", "must", "will", "shall"}
    features["modal_presence"] = int(any(w.lower() in modals for w in stem_words))

    # Quantitative flag
    features["quantitative_flag"] = int(any(c.isdigit() for c in full_text) or any(c in "+-*/=" for c in full_text))

    # Embedding similarity
    if _EMBED and embed_model and correct_text and distractors:
        emb_correct = embed_model.encode(correct_text)
        emb_distractors = embed_model.encode(distractors)
        similarities = [np.dot(emb_correct, emb_d) / (np.linalg.norm(emb_correct) * np.linalg.norm(emb_d))
                        for emb_d in emb_distractors]
        features["max_distractor_similarity"] = max(similarities) if similarities else 0.0
        features["avg_distractor_similarity"] = np.mean(similarities) if similarities else 0.0
    else:
        features["max_distractor_similarity"] = 0.0
        features["avg_distractor_similarity"] = 0.0

    # ---------- Ensure JSON‑safe types ----------
    for k, v in features.items():
        if isinstance(v, np.floating):
            features[k] = float(v)
        elif isinstance(v, np.integer):
            features[k] = int(v)
        elif isinstance(v, np.ndarray):
            features[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            features[k] = float(v)
    return features