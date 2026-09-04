import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
CSV_PATH = '/home/sujal/Programming/MajorProject/evaluation/all_features_combined.csv'
OUTPUT_DIR = '/home/sujal/Programming/MajorProject/evaluation'
# -------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

# ---- 1. Self-reported difficulty ----
# Convert to numeric: Easy=1, Medium=2, Hard=3
difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
df['self_reported_num'] = df['difficulty'].map(difficulty_map)

# ---- 2. Compute Empirical Difficulty from features ----
# You can adjust weights based on intuition or use a regression model.
# Here's a simple heuristic:
#   - Longer stems -> harder
#   - Longer options -> easier? (or harder? adjust)
#   - Higher distractor similarity -> harder
#   - Higher explanation length -> easier (more help)
#   - Lower NIR -> easier (less new info)
#   - Lower Flesch-Kincaid -> harder (complex text)

# Normalise features to 0-1 range for fair weighting
features = [
    'stem_word_count',
    'avg_option_word_count',
    'avg_distractor_jaccard_to_correct',
    'explanation_word_count',
    'expl_new_info_ratio',
    'flesch_kincaid_grade'
]

# Normalise each feature (min-max scaling)
df_norm = df.copy()
for f in features:
    min_val = df[f].min()
    max_val = df[f].max()
    if max_val - min_val > 0:
        df_norm[f] = (df[f] - min_val) / (max_val - min_val)
    else:
        df_norm[f] = 0.5

# Weighted sum (weights chosen empirically – you can tune these)
weights = {
    'stem_word_count': 0.30,               # longer = harder
    'avg_option_word_count': -0.10,        # longer options = easier? (or adjust)
    'avg_distractor_jaccard_to_correct': 0.25,  # more similar = harder
    'explanation_word_count': -0.10,       # longer explanation = easier (more help)
    'expl_new_info_ratio': 0.20,           # more new info = harder (complex)
    'flesch_kincaid_grade': 0.25           # higher grade = harder
}

df['empirical_score'] = sum(df_norm[f] * weights[f] for f in features)

# Scale empirical score to 1-3 range (Easy=1, Medium=2, Hard=3)
min_score = df['empirical_score'].min()
max_score = df['empirical_score'].max()
if max_score - min_score > 0:
    df['empirical_score_scaled'] = 1 + (df['empirical_score'] - min_score) / (max_score - min_score) * 2
else:
    df['empirical_score_scaled'] = 2

# Bin into categories (using quantiles for balanced distribution)
# This ensures roughly equal numbers in each category.
q1 = df['empirical_score_scaled'].quantile(0.33)
q2 = df['empirical_score_scaled'].quantile(0.67)

def bin_difficulty(score):
    if score <= q1:
        return 'Easy'
    elif score <= q2:
        return 'Medium'
    else:
        return 'Hard'

df['empirical_category'] = df['empirical_score_scaled'].apply(bin_difficulty)
df['empirical_num'] = df['empirical_category'].map(difficulty_map)

# ---- 3. Agreement Metrics ----
# Accuracy
accuracy = accuracy_score(df['self_reported_num'], df['empirical_num'])
print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

# Cohen's Kappa (unweighted)
kappa = cohen_kappa_score(df['self_reported_num'], df['empirical_num'])
print(f"Cohen's Kappa: {kappa:.3f}")

# Weighted Kappa (ordinal - penalizes larger disagreements more)
# Use linear weights: |i-j|
from sklearn.metrics import cohen_kappa_score
weighted_kappa = cohen_kappa_score(df['self_reported_num'], df['empirical_num'], weights='linear')
print(f"Weighted Kappa (linear): {weighted_kappa:.3f}")

# Spearman correlation
corr, pval = spearmanr(df['self_reported_num'], df['empirical_num'])
print(f"Spearman correlation: {corr:.3f} (p={pval:.4f})")

# ---- 4. Confusion Matrix ----
cm = confusion_matrix(df['self_reported_num'], df['empirical_num'])
print("\nConfusion Matrix (rows=True, cols=Predicted):")
print(pd.DataFrame(cm, index=['Easy', 'Medium', 'Hard'], columns=['Easy', 'Medium', 'Hard']))

# ---- 5. Per-Subject Agreement (optional) ----
print("\n--- Per-Subject Agreement ---")
for sub in df['subject'].unique():
    sub_df = df[df['subject'] == sub]
    acc = accuracy_score(sub_df['self_reported_num'], sub_df['empirical_num'])
    k = cohen_kappa_score(sub_df['self_reported_num'], sub_df['empirical_num'])
    wk = cohen_kappa_score(sub_df['self_reported_num'], sub_df['empirical_num'], weights='linear')
    print(f"{sub}: Acc={acc:.3f}, Kappa={k:.3f}, Weighted Kappa={wk:.3f}")

# ---- 6. Feature Contribution (correlation with empirical score) ----
print("\n--- Feature Contribution to Empirical Difficulty ---")
for f in features:
    corr = df[f].corr(df['empirical_score'])
    print(f"{f}: r = {corr:.3f}")

# ---- 7. Visualise Confusion Matrix ----
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Easy', 'Medium', 'Hard'],
            yticklabels=['Easy', 'Medium', 'Hard'])
plt.xlabel('Predicted (Empirical)')
plt.ylabel('True (Self-Reported)')
plt.title('Confusion Matrix: Self-Reported vs Empirical Difficulty')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/difficulty_confusion_matrix.png', dpi=300)
print(f"\n✅ Confusion matrix saved to: {OUTPUT_DIR}/difficulty_confusion_matrix.png")

# ---- 8. Save results ----
# Save the full DataFrame with empirical scores for later use
df.to_csv(f'{OUTPUT_DIR}/difficulty_analysis.csv', index=False)
print(f"✅ Detailed difficulty analysis saved to: {OUTPUT_DIR}/difficulty_analysis.csv")