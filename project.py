import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.utils import resample

# ------------- Q1–2: Load and Prepare Data --------------------
df = pd.read_csv("data/Chandler.csv")  # Load data
df['Anchortype_num'] = df['Anchortype'].map({'round': 0, 'precise': 1})  # Encode precision
df_clean = df[df['DROP'] == 0].copy()  # Remove excluded participants

# ------------- Q3: Reconstruct Anchors from Janiszewski Materials -------------
anchor_values = {
    'pen': {0: 4.000, 1: 3.998},
    'Proteindrink': {0: 10.0, 1: 9.8},
    'lebron': {0: 0.500, 1: 0.498},
    'slidy': {0: 40.0, 1: 39.75},
    'Cheese': {0: 5.0, 1: 4.85},
    'Figurine': {0: 50.0, 1: 49.0},
    'TV': {0: 5000.0, 1: 4998.0},
    'beachhouse': {0: 800000.0, 1: 799800.0},
    'number': {0: 10000.0, 1: 9989.0}
}

# Calculate relative underestimation
rel_diffs = []
for item in anchor_values:
    anchor_col = f'{item}_anchor'
    df_clean[anchor_col] = df_clean['Anchortype_num'].map(anchor_values[item])
    rel_diff = (df_clean[anchor_col] - df_clean[item]) / df_clean[anchor_col]
    rel_diffs.append(rel_diff)

df_clean['mean_underestimation'] = np.nanmean(np.column_stack(rel_diffs), axis=1)

# ------------- Q4–5: Prepare Data for ANOVA and Check Assumptions -------------
anova_df = df_clean[['Participant', 'mean_underestimation', 'Anchortype_num', 'magnitude']].dropna()
anova_df['Anchortype'] = anova_df['Anchortype_num'].map({0: 'Round', 1: 'Precise'})
anova_df['Motivation'] = anova_df['magnitude'].map({0: 'Weak', 1: 'Strong'})

# Shapiro–Wilk (normality)
shapiro_results = anova_df.groupby(['Anchortype', 'Motivation'])['mean_underestimation'].apply(
    lambda x: shapiro(x)[1]).reset_index(name='Shapiro_p')

# Levene's test (homogeneity of variance)
levene_stat, levene_p = levene(
    *[group['mean_underestimation'].values for _, group in anova_df.groupby(['Anchortype', 'Motivation'])]
)

# ------------- Q6–7: Two-Way ANOVA and Effect Sizes -----------------
model = ols('mean_underestimation ~ C(Anchortype) * C(Motivation)', data=anova_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
eta_sq = anova_table['sum_sq'] / anova_table['sum_sq'].sum()  # eta squared = effect size

# ------------- Q7: Bar Plot with Confidence Intervals ----------------
group_stats = anova_df.groupby(['Anchortype', 'Motivation'])['mean_underestimation'].agg(['mean', 'std', 'count'])
group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])
group_stats['ci95'] = 1.96 * group_stats['se']
group_stats = group_stats.reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(data=group_stats, x='Anchortype', y='mean', hue='Motivation', ci=None, palette='Set2')
for i, row in group_stats.iterrows():
    plt.errorbar(x=i//2 + (i % 2 - 0.5) * 0.2, y=row['mean'], yerr=row['ci95'],
                 fmt='none', capsize=5, color='black')
plt.ylabel("Mean Relative Underestimation")
plt.title("Estimates Below Anchor by Anchor Precision and Motivation")
plt.tight_layout()
plt.show()

# ------------- Q6–7: Permutation Test for Robustness ----------------
def compute_f_stats(df):
    model = ols('mean_underestimation ~ C(Anchortype) * C(Motivation)', data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    return table['F'].iloc[:3].values

original_f_stats = compute_f_stats(anova_df)
n_permutations = 1000 # try with 5000 too, which will push the p value to borderline significance
f_dist = np.zeros((n_permutations, 3))
for i in range(n_permutations):
    shuffled_df = anova_df.copy()
    shuffled_df['mean_underestimation'] = resample(shuffled_df['mean_underestimation'].values)
    f_dist[i, :] = compute_f_stats(shuffled_df)

p_values_perm = (np.sum(f_dist >= original_f_stats, axis=0) + 1) / (n_permutations + 1)

# ------------- Q7: Permutation Distributions ----------------
effects = ['Anchor Precision', 'Motivation', 'Interaction']
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i in range(3):
    ax = axes[i]
    ax.hist(f_dist[:, i], bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(original_f_stats[i], color='red', linestyle='--', label=f'Observed F = {original_f_stats[i]:.3f}')
    ax.set_title(f'{effects[i]}\nPermutation Distribution')
    ax.set_xlabel('F value')
    ax.set_ylabel('Frequency')
    ax.legend()
plt.tight_layout()
plt.show()

# ------------- Q8–9: Interpretation (Narrative) ----------------
print("Interpretation:")
print("1. No statistically significant effects were found for anchor precision or motivation.")
print("2. Effect sizes were very small (eta squared < 0.01).")
print("3. Permutation tests confirmed the lack of significance. For n_parmutations = 5000, p-values were borderline significant. Close to p = 0.05")
print("4. Compared to Janiszewski & Uy (2008), this is a failed replication.")
print("5. The practical relevance of the precision effect is likely context-dependent.")
