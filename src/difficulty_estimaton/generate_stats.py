import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
CSV_PATH = '/home/sujal/Programming/MajorProject/evaluation/all_features_combined.csv'
# -------------------------------------------------------------------

# Load data
df = pd.read_csv(CSV_PATH)

# -------------------------------------------------------------------
# 1. Define metrics and pretty names
# -------------------------------------------------------------------
metrics = [
    'stem_word_count',
    'avg_option_word_count',
    'pairwise_avg_jaccard',
    'avg_distractor_jaccard_to_correct',
    'explanation_word_count',
    'expl_to_question_jaccard',
    'expl_to_correct_jaccard',
    'expl_new_info_ratio',
    'expl_contains_correct_verbatim',  # 0/1 -> will multiply by 100 for %
    'flesch_kincaid_grade'
]

pretty_names = {
    'stem_word_count': 'Stem words',
    'avg_option_word_count': 'Option words',
    'pairwise_avg_jaccard': 'Pairwise Jaccard',
    'avg_distractor_jaccard_to_correct': 'Distractor similarity',
    'explanation_word_count': 'Explanation words',
    'expl_to_question_jaccard': 'Expl→Question Jaccard',
    'expl_to_correct_jaccard': 'Expl→Correct Jaccard',
    'expl_new_info_ratio': 'New information ratio',
    'expl_contains_correct_verbatim': 'Verbatim copy (%)',
    'flesch_kincaid_grade': 'Flesch–Kincaid grade'
}

# Convert verbatim to percentage
df['expl_contains_correct_verbatim'] = df['expl_contains_correct_verbatim'] * 100

# -------------------------------------------------------------------
# 2. Mean ± Std per subject
# -------------------------------------------------------------------
subjects = sorted(df['subject'].unique())
subject_stats = {}
for sub in subjects:
    sub_df = df[df['subject'] == sub]
    subject_stats[sub] = {m: (sub_df[m].mean(), sub_df[m].std()) for m in metrics}

# Overall
overall_stats = {m: (df[m].mean(), df[m].std()) for m in metrics}

# -------------------------------------------------------------------
# 3. Print a nice table (console)
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("MEAN ± STD PER SUBJECT")
print("=" * 80)
print(f"{'Metric':<30}", end="")
for sub in subjects:
    print(f"{sub:>12}", end="")
print(f"{'Overall':>10}")
print("-" * 80)

for m in metrics:
    print(f"{pretty_names[m]:<30}", end="")
    for sub in subjects:
        mean, std = subject_stats[sub][m]
        print(f"{mean:>6.2f} ± {std:>4.2f}", end="")
    mean, std = overall_stats[m]
    print(f"{mean:>6.2f} ± {std:>4.2f}")
print("=" * 80)

# -------------------------------------------------------------------
# 4. Correlation matrix
# -------------------------------------------------------------------
corr_metrics = [
    'explanation_word_count',
    'expl_new_info_ratio',
    'expl_to_correct_jaccard',
    'expl_to_question_jaccard',
    'avg_option_word_count',
    'stem_word_count'
]
corr_df = df[corr_metrics].corr()

print("\n" + "=" * 80)
print("PEARSON CORRELATION MATRIX")
print("=" * 80)
# Replace column names with pretty names for readability
corr_df.columns = [pretty_names[c] for c in corr_df.columns]
corr_df.index = [pretty_names[c] for c in corr_df.index]
print(corr_df.round(3))
print("=" * 80)

# Extract key insights
print("\n🔍 Key Correlations:")
pairs = [
    ('explanation_word_count', 'expl_new_info_ratio'),
    ('explanation_word_count', 'expl_to_correct_jaccard'),
    ('expl_new_info_ratio', 'expl_to_correct_jaccard'),
    ('stem_word_count', 'explanation_word_count'),
]
for a, b in pairs:
    r = corr_df.loc[pretty_names[a], pretty_names[b]]
    print(f"  {pretty_names[a]} vs {pretty_names[b]}: r = {r:.3f}")

# -------------------------------------------------------------------
# 5. Generate LaTeX Table (copy & paste into your .tex file)
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("LATEX TABLE (copy & paste)")
print("=" * 80)
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Descriptive statistics of generated MCQs across subjects (mean $\\pm$ std)}")
print("\\label{tab:mcq_stats}")
print("\\begin{tabular}{l" + "c" * len(subjects) + "c}")
print("\\toprule")
print("\\textbf{Metric} & " + " & ".join(subjects) + " & \\textbf{Overall} \\\\")
print("\\midrule")

for m in metrics:
    row = pretty_names[m]
    for sub in subjects:
        mean, std = subject_stats[sub][m]
        # Use ± symbol; if std is very small, keep 2 decimals
        row += f" & {mean:.2f} $\\pm$ {std:.2f}"
    mean, std = overall_stats[m]
    row += f" & {mean:.2f} $\\pm$ {std:.2f}"
    row += " \\\\"
    print(row)

# Add N row (counts)
counts = [str(len(df[df['subject'] == sub])) for sub in subjects]
print(f"N (MCQs) & " + " & ".join(counts) + f" & {len(df)} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
print("=" * 80)