import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load original data with QUOTE_NONE to prevent unescaped quotation marks
# from merging lines. This ensures exactly 5,574 messages (4,827 ham, 747 spam).
raw_path = 'data/SMSSpamCollection'
df = pd.read_csv(raw_path, sep='\t', header=None, names=['label', 'text'], quoting=csv.QUOTE_NONE, encoding='utf-8')

# Output directories: save to both 'data_splits/' and 'data/data_splits/'
base_dirs = ['data_splits', os.path.join('data', 'data_splits')]

for base_dir in base_dirs:
    os.makedirs(os.path.join(base_dir, 'with_duplicates'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'no_duplicates'), exist_ok=True)

# -------------------------------------------------------------
# Split 1: WITH DUPLICATES (5,574 total: 4,827 ham, 747 spam)
# 70% Train, 15% Val, 15% Test (stratified by label, seed=42)
# -------------------------------------------------------------
train_with, temp_with = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df['label']
)
val_with, test_with = train_test_split(
    temp_with, test_size=0.50, random_state=42, stratify=temp_with['label']
)

# -------------------------------------------------------------
# Split 2: NO DUPLICATES (5,171 total: 4,518 ham, 653 spam)
# 70% Train, 15% Val, 15% Test (stratified by label, seed=42)
# -------------------------------------------------------------
df_no_dup = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)

train_no, temp_no = train_test_split(
    df_no_dup, test_size=0.30, random_state=42, stratify=df_no_dup['label']
)
val_no, test_no = train_test_split(
    temp_no, test_size=0.50, random_state=42, stratify=temp_no['label']
)

# Save to both directory locations
for base_dir in base_dirs:
    # Save with_duplicates
    with_dir = os.path.join(base_dir, 'with_duplicates')
    train_with.to_csv(os.path.join(with_dir, 'train.csv'), index=False)
    val_with.to_csv(os.path.join(with_dir, 'val.csv'), index=False)
    test_with.to_csv(os.path.join(with_dir, 'test.csv'), index=False)

    # Save no_duplicates
    no_dir = os.path.join(base_dir, 'no_duplicates')
    train_no.to_csv(os.path.join(no_dir, 'train.csv'), index=False)
    val_no.to_csv(os.path.join(no_dir, 'val.csv'), index=False)
    test_no.to_csv(os.path.join(no_dir, 'test.csv'), index=False)

# ==============================================================================
# MAIN CHECKS: CLASS COUNTS & DUPLICATES VERIFICATION
# ==============================================================================

def print_summary(name, train, val, test, total_len):
    print(f"=== CHECK 1: CLASS DISTRIBUTION - {name} (Total: {total_len}) ===")
    for split_name, split_df in [('Train (70%)', train), ('Val (15%)', val), ('Test (15%)', test)]:
        counts = split_df['label'].value_counts()
        pcts = split_df['label'].value_counts(normalize=True) * 100
        print(f"  {split_name:12s}: Total={len(split_df):4d} | Ham={counts['ham']:4d} ({pcts['ham']:.2f}%) | Spam={counts['spam']:3d} ({pcts['spam']:.2f}%)")
    print()

print_summary("WITH DUPLICATES", train_with, val_with, test_with, len(df))
print_summary("NO DUPLICATES", train_no, val_no, test_no, len(df_no_dup))

print("=== CHECK 2: DUPLICATES & CROSS-SPLIT LEAKAGE ===")
train_with_set = set(train_with['text'])
val_with_leak = val_with['text'].isin(train_with_set).sum()
test_with_leak = test_with['text'].isin(train_with_set).sum()
print("WITH DUPLICATES:")
print(f"  Internal duplicates in Train: {train_with.duplicated(subset=['text']).sum()}")
print(f"  Internal duplicates in Val  : {val_with.duplicated(subset=['text']).sum()}")
print(f"  Internal duplicates in Test : {test_with.duplicated(subset=['text']).sum()}")
print(f"  Val messages leaked from Train : {val_with_leak} ({val_with_leak / len(val_with) * 100:.2f}%)")
print(f"  Test messages leaked from Train: {test_with_leak} ({test_with_leak / len(test_with) * 100:.2f}%)")

train_no_set = set(train_no['text'])
val_no_leak = val_no['text'].isin(train_no_set).sum()
test_no_leak = test_no['text'].isin(train_no_set).sum()
print("\nNO DUPLICATES:")
print(f"  Internal duplicates in Train: {train_no.duplicated(subset=['text']).sum()}")
print(f"  Internal duplicates in Val  : {val_no.duplicated(subset=['text']).sum()}")
print(f"  Internal duplicates in Test : {test_no.duplicated(subset=['text']).sum()}")
print(f"  Val messages leaked from Train : {val_no_leak} (0.00%)")
print(f"  Test messages leaked from Train: {test_no_leak} (0.00%)")
