"""Load and validate the frozen SMS Spam Collection splits."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "SMSSpamCollection"
SPLITS_DIR = PROJECT_ROOT / "data_splits"
VARIANTS = ("with_duplicates", "no_duplicates")
RECORD_COLUMNS = ["record_id", "label", "text"]
LABELS = {"ham", "spam"}


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame | None = None

    def items(self) -> Iterator[tuple[str, pd.DataFrame]]:
        yield "train", self.train
        yield "val", self.validation
        if self.test is not None:
            yield "test", self.test


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_records(records: pd.DataFrame, *, context: str = "data") -> None:
    missing = set(RECORD_COLUMNS) - set(records.columns)
    if missing:
        raise ValueError(f"{context}: missing columns {sorted(missing)}.")
    if records.empty:
        raise ValueError(f"{context}: no records found.")

    for column in RECORD_COLUMNS:
        if not records[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{context}: {column} must contain strings without missing values.")

    if not records["record_id"].str.fullmatch(r"sms_[0-9]{6,}").all():
        raise ValueError(f"{context}: invalid record IDs.")
    if records["record_id"].duplicated().any():
        raise ValueError(f"{context}: duplicate record IDs.")
    if not records["label"].isin(LABELS).all():
        raise ValueError(f"{context}: labels must be ham or spam.")
    if records["text"].str.strip().eq("").any():
        raise ValueError(f"{context}: empty or whitespace-only messages.")
    if records.groupby("text")["label"].nunique().gt(1).any():
        raise ValueError(f"{context}: identical messages have conflicting labels.")


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Assign IDs by source line number without modifying message text."""
    records = []
    with Path(path).open(encoding="utf-8", newline="") as source:
        for line_number, line in enumerate(source, start=1):
            label, separator, text = line.removesuffix("\n").removesuffix("\r").partition("\t")
            if not separator:
                raise ValueError(f"{path}: missing tab separator on line {line_number}.")
            records.append((f"sms_{line_number:06d}", label, text))

    frame = pd.DataFrame(records, columns=RECORD_COLUMNS)
    validate_records(frame, context=str(path))
    return frame


def validate_splits(splits: DataSplits, *, variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose one of {VARIANTS}.")
    for name, records in splits.items():
        validate_records(records, context=f"{variant}/{name}")
        if set(records["label"]) != LABELS:
            raise ValueError(f"{variant}/{name}: both classes must be present.")

    combined = pd.concat([records for _, records in splits.items()], ignore_index=True)
    validate_records(combined, context=variant)
    if variant == "no_duplicates" and combined["text"].duplicated().any():
        raise ValueError("no_duplicates: identical messages found within or across splits.")


def load_splits(
    variant: str = "no_duplicates",
    *,
    split_dir: str | Path = SPLITS_DIR,
    include_test: bool = False,
) -> DataSplits:
    """Load frozen files and verify their manifest; test access is explicit."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose one of {VARIANTS}.")

    split_dir = Path(split_dir)
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}. Run create_splits.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{manifest_path}: unsupported manifest schema.")

    names = ("train", "val", "test") if include_test else ("train", "val")
    frames = {}
    for name in names:
        path = split_dir / variant / f"{name}.csv"
        try:
            expected = manifest["variants"][variant][name]
            expected_hash = expected["sha256"]
            expected_count = expected["record_count"]
            expected_classes = expected["class_counts"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{manifest_path}: missing metadata for {variant}/{name}.") from error

        if file_sha256(path) != expected_hash:
            raise ValueError(f"{path}: checksum differs from the frozen manifest.")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        validate_records(frame, context=str(path))
        actual_classes = frame["label"].value_counts().to_dict()
        if len(frame) != expected_count or actual_classes != expected_classes:
            raise ValueError(f"{path}: record counts differ from the frozen manifest.")
        frames[name] = frame

    splits = DataSplits(frames["train"], frames["val"], frames.get("test"))
    validate_splits(splits, variant=variant)
    return splits
