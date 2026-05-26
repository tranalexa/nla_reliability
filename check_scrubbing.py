import pandas as pd
import re

df = pd.read_csv('data/selfdescribe_400.csv')

# Check for Dr. followed by a name
dr_hits = df['user_prompt'].str.contains(r'Dr\.\s+[A-Z][a-z]+', regex=True).sum()
print(f'Rows with Dr. + name: {dr_hits}/{len(df)}')

# Check for any remaining pronouns
pronoun_hits = df['user_prompt'].str.contains(r'\b(he|she|him|her|his|hers|himself|herself)\b', case=False, regex=True).sum()
print(f'Rows with pronouns remaining: {pronoun_hits}/{len(df)}')

# Check for Mr/Mrs/Ms
honorific_hits = df['user_prompt'].str.contains(r'\b(Mr|Mrs|Ms)\.?\b', regex=True).sum()
print(f'Rows with honorifics remaining: {honorific_hits}/{len(df)}')

# Show a few examples with names still present
examples = df[df['user_prompt'].str.contains(r'Dr\.\s+[A-Z][a-z]+', regex=True)]['user_prompt'].head(3)
print('\nExamples with names:')
for ex in examples:
    print(' ', ex[:200])
    print()