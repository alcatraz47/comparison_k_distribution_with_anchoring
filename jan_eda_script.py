import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.utils import resample
from scipy.stats import zscore

df = pd.read_csv("data/Chandler.csv")
df_clean = df[df["DROP"] == 0].copy()
df_clean["Anchortype_num"] = df_clean["Anchortype"].map({"round": 0, "precise": 1})
anchors = {
    "pen": {0: 4.000, 1: 3.998},
    "Proteindrink": {0: 10.0, 1: 9.8},
    "lebron": {0: 0.500, 1: 0.498},
    "slidy": {0: 40.00, 1: 39.75},
    "Cheese": {0: 5.00, 1: 4.85},
    "Figurine": {0: 50.00, 1: 49.00},
    "TV": {0: 5000.00, 1: 4998.00},
    "beachhouse": {0: 800000.00, 1: 799800.00},
    "number": {0: 10000, 1: 9989},
}
for item in anchors:
    df_clean[item + "_dev"] = df_clean.apply(
        lambda row: row[item] - anchors[item][row["Anchortype_num"]], axis=1
    )
    df_clean["Z_" + item] = zscore(df_clean[item + "_dev"])

z_columns = [f"Z_{item}" for item in anchors]
df_clean["Zmean"] = df_clean[z_columns].mean(axis=1)
# ------------- Q4: Check Assumptions for ANOVA --------------------
anova_model = ols("Zmean ~ C(Anchortype) * C(magnitude)", data=df_clean).fit()
anova_results = sm.stats.anova_lm(anova_model, typ=2)
print(f"[Info:] ANOVA Two Way Result: \n{anova_results}")

# Normality
shapiro_test = shapiro(anova_model.resid)
print(f"[Info:] Shapiro-Wilk test:\n{shapiro_test}")

# Homogeneity of variances
levene_test = levene(
    df_clean[df_clean["Anchortype"] == "round"]["Zmean"],
    df_clean[df_clean["Anchortype"] == "precise"]["Zmean"]
)
print(f"[Info:] Levene's test:\n{levene_test}")

ss_total = sum(anova_results["sum_sq"])
anova_results["eta_sq"] = anova_results["sum_sq"] / ss_total
print(f"Eta squared effect sizes:\n{anova_results['eta_sq']}")

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean, x="Anchortype", y="Zmean", hue="magnitude")
plt.title("Zmean by Anchor Precision and Motivation")
plt.xlabel("Anchor Type (Precision)")
plt.ylabel("Mean Standardized Deviation (Zmean)")
plt.legend(title="Motivation")
plt.tight_layout()
plt.show()

print("ANOVA Results:\n", anova_results)
print("\nShapiro-Wilk Test (Normality):\n", shapiro_test)
print("\nLevene’s Test (Homogeneity of Variance):\n", levene_test)