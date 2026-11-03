"""Inspect duplicates without modifying splits or model input text."""

import argparse
import json
import unicodedata
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sms_spam.data import (
    SPLITS_DIR,
    VARIANTS,
    DataSplits,
    file_sha256,
    load_splits,
    validate_records,
)

NEAR_DUPLICATE_COLUMNS = [
    "left_id", "right_id", "left_split", "right_split", "similarity"
]


@dataclass(frozen=True)
class DuplicateAudit:
    within_splits: pd.DataFrame
    between_splits: pd.DataFrame
    near_duplicates: pd.DataFrame
    parameters: dict[str, object]


def normalize_for_audit(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def audit_duplicates(
    splits: DataSplits,
    *,
    similarity_threshold: float = 0.90,
    min_length: int = 40,
    min_length_ratio: float = 0.80,
    batch_size: int = 256,
) -> DuplicateAudit:
    """Find cross-split candidates using audit-only character TF-IDF cosine scores.

    Normalized equality is reported separately. Similarity matches are candidates,
    not confirmed duplicates. The audit vectorizer is never used by a model.
    """
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in (0, 1].")
    if not 0 < min_length_ratio <= 1:
        raise ValueError("min_length_ratio must be in (0, 1].")
    if min_length < 3 or batch_size < 1:
        raise ValueError("min_length must be at least 3 and batch_size must be positive.")

    frames = {}
    within_rows = []
    for name, frame in splits.items():
        validate_records(frame, context=name)
        normalized = frame["text"].map(normalize_for_audit)
        frames[name] = frame.assign(normalized_text=normalized, split=name)
        within_rows.append({
            "split": name,
            "records": len(frame),
            "exact_duplicate_rows": int(frame["text"].duplicated().sum()),
            "normalized_duplicate_rows": int(normalized.duplicated().sum()),
        })

    combined = pd.concat(frames.values(), ignore_index=True)
    validate_records(combined, context="duplicate audit")
    between_rows = []
    for left_name, right_name in combinations(frames, 2):
        left, right = frames[left_name], frames[right_name]
        exact = set(left["text"]) & set(right["text"])
        normalized = set(left["normalized_text"]) & set(right["normalized_text"])
        between_rows.append({
            "left_split": left_name,
            "right_split": right_name,
            "shared_exact_texts": len(exact),
            "left_exact_overlap_rows": int(left["text"].isin(exact).sum()),
            "right_exact_overlap_rows": int(right["text"].isin(exact).sum()),
            "shared_normalized_texts": len(normalized),
        })

    eligible = combined.loc[
        combined["normalized_text"].str.len().ge(min_length)
    ].reset_index(drop=True)
    candidates = []
    if len(eligible) > 1:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False)
        vectors = vectorizer.fit_transform(eligible["normalized_text"])
        texts = eligible["normalized_text"].tolist()
        record_ids = eligible["record_id"].tolist()
        split_names = eligible["split"].tolist()

        for start in range(0, len(eligible), batch_size):
            similarities = (vectors[start:start + batch_size] @ vectors.T).tocoo()
            selected = similarities.data >= similarity_threshold
            for row, column, similarity in zip(
                similarities.row[selected], similarities.col[selected], similarities.data[selected]
            ):
                left, right = start + row, column
                if left >= right or split_names[left] == split_names[right]:
                    continue
                if texts[left] == texts[right]:
                    continue
                lengths = (len(texts[left]), len(texts[right]))
                if min(lengths) / max(lengths) < min_length_ratio:
                    continue
                candidates.append((
                    record_ids[left], record_ids[right],
                    split_names[left], split_names[right], float(similarity),
                ))

    near_duplicates = pd.DataFrame(candidates, columns=NEAR_DUPLICATE_COLUMNS)
    if not near_duplicates.empty:
        near_duplicates = near_duplicates.sort_values(
            ["left_id", "right_id"]
        ).reset_index(drop=True)
    parameters = {
        "normalization": "NFKC, casefold, whitespace collapse",
        "representation": "character TF-IDF fitted only for this audit",
        "ngram_range": [3, 5],
        "similarity": "cosine",
        "similarity_threshold": similarity_threshold,
        "min_length": min_length,
        "min_length_ratio": min_length_ratio,
        "normalized_equal_pairs_excluded": True,
    }
    return DuplicateAudit(
        pd.DataFrame(within_rows), pd.DataFrame(between_rows), near_duplicates, parameters
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="no_duplicates")
    parser.add_argument("--split-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    splits = load_splits(arguments.variant, split_dir=arguments.split_dir, include_test=True)
    audit = audit_duplicates(splits)
    print(audit.within_splits.to_string(index=False))
    print(audit.between_splits.to_string(index=False))
    print(f"Additional near-duplicate candidate pairs: {len(audit.near_duplicates)}")

    if arguments.output is not None:
        report = {
            "variant": arguments.variant,
            "split_manifest_sha256": file_sha256(arguments.split_dir / "manifest.json"),
            "parameters": audit.parameters,
            "within_splits": audit.within_splits.to_dict(orient="records"),
            "between_splits": audit.between_splits.to_dict(orient="records"),
            "near_duplicate_candidates": audit.near_duplicates.to_dict(orient="records"),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
