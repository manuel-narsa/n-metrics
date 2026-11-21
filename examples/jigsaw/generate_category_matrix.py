import pandas as pd
import numpy as np
import argparse
import os
import sys
from itertools import chain
from pathlib import Path

# ==============================================================================
# ARGUMENT CONFIGURATION
# ==============================================================================
parser = argparse.ArgumentParser(description="Generates a matrix for the N-Framework using pseudo-annotators by toxicity subtype.")
parser.add_argument("--file", type=str, default="toxicity_individual_annotations.csv", help="Path to the input CSV file (relative to project root)")
parser.add_argument("--min", type=int, default=3, help="Minimum number of annotators who indicated toxicity per subject")
parser.add_argument("--max", type=int, default=15, help="Maximum number of annotators who indicated toxicity per subject")
parser.add_argument("--output", type=str, default="toxicity_pseudo_annotators_matrix.csv", help="Name of the output CSV file (relative to project root)")

# parse_known_args() allows it to work inside Jupyter/IPython
args, _ = parser.parse_known_args()

# ==============================================================================
# PATH CONFIGURATION (RELATIVE TO PROJECT ROOT)
# ==============================================================================
# Define the root directory of the project.
# If the script is in the root, this is its directory.
# If the script is in a subfolder (e.g., /scripts), uncomment the line below to go up one level.
BASE_DIR = Path(__file__).resolve().parent
# BASE_DIR = BASE_DIR.parent  # Uncomment if the script is inside a subfolder

# Build absolute paths based on the project root
INPUT_PATH = BASE_DIR / args.file
OUTPUT_PATH = BASE_DIR / args.output

# ==============================================================================
# 1. DATA LOADING AND STRICT FILTERING
# ==============================================================================
print(f"Reading {INPUT_PATH.name}...")
if not INPUT_PATH.exists():
    print(f"❌ Error: Input file not found at {INPUT_PATH}")
    sys.exit(1)

df_columns = pd.read_csv(INPUT_PATH, nrows=0)
all_columns = list(df_columns.columns)

# Main label columns (toxicity type)
label_columns = ['toxic', 'severe_toxic']
# Subtype columns (to be used as pseudo-annotators)
subtype_columns = ['identity_attack', 'insult', 'obscene', 'sexual_explicit', 'threat']

# Check that all required columns exist
missing = [c for c in label_columns + subtype_columns + ['id', 'worker'] if c not in all_columns]
if missing:
    print(f"❌ Error: Missing columns in the CSV: {missing}")
    sys.exit(1)

columns_to_use = ['id', 'worker'] + label_columns + subtype_columns
df = pd.read_csv(
    INPUT_PATH,
    usecols=columns_to_use,
    dtype={c: 'int8' for c in label_columns + subtype_columns}
)

print("Filtering subjects with at least 2 annotators who indicated toxicity...")

# An annotator "marks" if they put 1 in toxic or severe_toxic
df['is_toxic'] = ((df['toxic'] == 1) | (df['severe_toxic'] == 1)).astype('int8')

# Keep only the evaluations where the annotator considered the comment toxic
df_toxic = df[df['is_toxic'] == 1].copy()

# Count of annotators who indicated toxicity per ID
toxic_annotators_per_id = df_toxic.groupby('id')['worker'].nunique()

# 1) At least 2 annotators who indicated toxicity
valid_ids = toxic_annotators_per_id[toxic_annotators_per_id >= 2].index

# 2) Range --min/--max applied ONLY to annotators who indicated toxicity
range_ids = toxic_annotators_per_id[
    (toxic_annotators_per_id >= args.min) &
    (toxic_annotators_per_id <= args.max)
].index

# Intersection of both criteria
final_ids = valid_ids.intersection(range_ids)

df_filtered = df_toxic[df_toxic['id'].isin(final_ids)].copy()

print(f"Final valid subjects: {df_filtered['id'].nunique():,}")

# ==============================================================================
# 2. UNFOLDING TO PSEUDO-ANNOTATORS (SUBTYPES + CATEGORY 0)
# ==============================================================================
print("Unfolding annotators into pseudo-annotators by subtype...")

# Subtype mapping to integer (1–5). The value 0 is reserved for
# "toxic without specified subtype".
subtype_map = {
    'identity_attack': 1,
    'insult': 2,
    'obscene': 3,
    'sexual_explicit': 4,
    'threat': 5,
}
VALUE_WITHOUT_SUBTYPE = 0

def extract_pseudo_annotators(row):
    """
    Returns a tuple with the integers corresponding to the subtypes
    the annotator marked with 1. If no subtype was marked, returns (0,)
    to indicate "toxic without specified subtype".
    """
    indices = []
    for col, value in subtype_map.items():
        if row[col] == 1:
            indices.append(value)
    if not indices:
        # Toxic but without subtype: we encode it as 0
        return (VALUE_WITHOUT_SUBTYPE,)
    return tuple(indices)

# Apply the function to each row (only annotators who indicated toxicity remain)
df_filtered['pseudo_annotators'] = df_filtered.apply(extract_pseudo_annotators, axis=1)

# We no longer remove empty rows: all have at least one value (0 if no subtype)

# Informative statistic: how many annotators were encoded as "without subtype"
n_without_subtype = (df_filtered['pseudo_annotators'].apply(lambda t: t == (VALUE_WITHOUT_SUBTYPE,))).sum()
print(f"Annotators encoded as 'toxic without subtype' (value 0): {n_without_subtype:,}")

print("Collapsing the matrix...")
# Group by ID and concatenate all tuples into a single list
collapsed_profiles = df_filtered.groupby('id')['pseudo_annotators'].apply(
    lambda x: list(chain.from_iterable(x))
)

# Convert to DataFrame (shorter rows will be padded with NaN)
df_matrix = pd.DataFrame(collapsed_profiles.tolist())

# Force to integer (Int64) so Streamlit treats it as Nominal
df_matrix = df_matrix.astype('Int64')

# Save
df_matrix.to_csv(OUTPUT_PATH, index=False, header=False)

print(f"\n✅ Matrix saved to {OUTPUT_PATH}!")
print(f"   Dimensions: {df_matrix.shape[0]} rows x {df_matrix.shape[1]} maximum pseudo-annotators.")
print(f"   The integer values represent:")
print(f"     0: toxic without specified subtype")
for name, value in subtype_map.items():
    print(f"     {value}: {name}")

# ==============================================================================
# 3. USAGE
# ==============================================================================
# The basic command is python script_name.py. Depending on what you want to analyze, add the parameters. Here are the most useful cases:

# Case 1: Use default values (Range 3 to 15 annotators, all toxic)
# If you add nothing else, it will use the default input file and filters (minimum 3 annotators, maximum 15, and will require at least 2 annotators who marked something):

# powershell
# python generate_pseudo_matrix.py
# Case 2: Change the annotator range (e.g., from 5 to 20)

# powershell
# python generate_pseudo_matrix.py --min 5 --max 20
# Case 3: If your CSV file is in another folder (relative to the project root)

# powershell
# python generate_pseudo_matrix.py --file "data/toxicity_individual_annotations.csv"
# Case 4: Change the output file name

# powershell
# python generate_pseudo_matrix.py --output "my_final_matrix.csv"
# Case 5: Combine multiple parameters

# powershell
# python generate_pseudo_matrix.py --min 3 --max 15 --output "matrix_15_annotators.csv"
# 3. What you will see on the screen?
# While it runs, you will see progress messages in the terminal:

# "Reading toxicity_individual_annotations.csv..."

# "Filtering subjects with at least 2 real annotators..."

# "Final valid subjects: [number]..."

# "Unfolding annotators into pseudo-annotators by category..."

# "✅ Matrix saved to pseudo_annotators_matrix.csv!"