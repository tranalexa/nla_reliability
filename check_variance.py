import pandas as pd
import numpy as np

pc = pd.read_parquet("data/pairwise_consistency.parquet")

# Within-activation variance of cosine similarities
within_var = pc.groupby("activation_idx")["cos_sim"].var()
print("Mean within-activation cos_sim variance:", within_var.mean())
print("Min:", within_var.min())
print("Max:", within_var.max())

# Distribution of mean cos_sim per activation
within_mean = pc.groupby("activation_idx")["cos_sim"].mean()
print("\nMean cos_sim per activation:")
print(within_mean.describe())