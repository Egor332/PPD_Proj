"""Reproduce the milestone 1 splits and attach source record identifiers."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sms_spam.data import (
    RAW_DATA_PATH,
    SPLITS_DIR,
    DataSplits,
    file_sha256,
    load_raw_data,
    validate_splits,
)

SPLIT_SEED = 42


def split_records(records: pd.DataFrame) -> DataSplits:
    train, remaining = train_test_split(
        records, test_size=0.30, random_state=SPLIT_SEED, stratify=records["label"]
    )
    validation, test = train_test_split(
        remaining, test_size=0.50, random_state=SPLIT_SEED, stratify=remaining["label"]
    )
    return DataSplits(*(frame.reset_index(drop=True) for frame in (train, validation, test)))


def prepare_splits(
    raw_path: str | Path = RAW_DATA_PATH,
    output_dir: str | Path = SPLITS_DIR,
) -> dict[str, DataSplits]:
    """Write reproducible splits, refusing to change existing record assignments."""
    records = load_raw_data(raw_path)
    variants = {
        "with_duplicates": split_records(records),
        "no_duplicates": split_records(records.drop_duplicates(subset="text", keep="first")),
    }
    output_dir = Path(output_dir)
    source_hash = file_sha256(raw_path)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("dataset", {}).get("sha256") != source_hash:
            raise ValueError(
                "Source data changed. Use a new output directory for a new dataset version."
            )

    for variant, splits in variants.items():
        validate_splits(splits, variant=variant)
        for name, frame in splits.items():
            path = output_dir / variant / f"{name}.csv"
            if path.exists():
                existing = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
                columns = (
                    list(frame.columns) if "record_id" in existing.columns else ["label", "text"]
                )
                matching_columns = set(columns) == set(existing.columns)
                if not matching_columns or not existing[columns].equals(frame[columns]):
                    raise ValueError(f"{path}: refusing to change existing records or their order.")

    manifest = {
        "schema_version": 1,
        "dataset": {
            "name": "SMS Spam Collection v.1",
            "sha256": source_hash,
            "record_count": len(records),
            "record_id": "sms_ followed by the one-based source line number, padded to six digits",
        },
        "split": {
            "seed": SPLIT_SEED,
            "stratify": "label",
            "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "deduplication": "no_duplicates only: keep the first occurrence of each exact text",
        },
        "variants": {},
    }
    for variant, splits in variants.items():
        manifest["variants"][variant] = {}
        for name, frame in splits.items():
            path = output_dir / variant / f"{name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
            manifest["variants"][variant][name] = {
                "record_count": len(frame),
                "class_counts": frame["label"].value_counts().to_dict(),
                "sha256": file_sha256(path),
            }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR)
    arguments = parser.parse_args()
    variants = prepare_splits(arguments.raw_path, arguments.output_dir)
    rows = [
        {
            "variant": variant, "split": name, "records": len(frame),
            **frame["label"].value_counts().to_dict(),
        }
        for variant, splits in variants.items()
        for name, frame in splits.items()
    ]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
