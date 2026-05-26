import pandas as pd

desc = pd.read_parquet("data/descriptions.parquet")
pc   = pd.read_parquet("data/pairwise_consistency.parquet")
fs   = pd.read_parquet("data/fidelity_scores.parquet")

print("descriptions:", desc.shape)   # expect (4800, ...)
print("pairwise:    ", pc.shape)     # expect (26400, ...)
print("fidelity:    ", fs.shape)     # expect (4800, ...)
print()
print(desc.head(2))
print()
print(pc.head(2))
print()
print(fs.head(2))