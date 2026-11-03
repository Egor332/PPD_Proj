"""Train text classifiers on frozen training data and evaluate on validation data."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from sms_spam.config import DEFAULT_CONFIG_PATH, ExperimentConfig, load_config
from sms_spam.data import PROJECT_ROOT, SPLITS_DIR, VARIANTS, file_sha256, load_splits
from sms_spam.evaluation import evaluate_predictions, predict_records
from sms_spam.models import BASELINE_NAMES, MODEL_NAMES, BaselineModel, build_model
from sms_spam.preprocessing import preprocessing_config, preprocessing_summary

RESULTS_DIR = PROJECT_ROOT / "reports" / "baselines"
PREPROCESSING_RESULTS_DIR = PROJECT_ROOT / "reports" / "preprocessing"


@dataclass(frozen=True)
class ExperimentResult:
    model_name: str
    dataset_variant: str
    model: BaselineModel
    predictions: pd.DataFrame
    metrics: dict[str, object]
    train_seconds: float
    inference_seconds: float
    config: dict[str, object]
    metadata: dict[str, object]
    output_dir: Path | None

    def summary(self) -> dict[str, object]:
        scalar_metrics = {
            name: value for name, value in self.metrics.items() if isinstance(value, (int, float))
        }
        return {
            "dataset_variant": self.dataset_variant,
            "model": self.model_name,
            **scalar_metrics,
            "train_seconds": self.train_seconds,
            "inference_seconds": self.inference_seconds,
            "inference_ms_per_message": 1000 * self.inference_seconds / len(self.predictions),
            "run_id": self.metadata["run_id"],
        }


def _code_metadata() -> dict[str, object]:
    digest = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "sms_spam").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())

    metadata = {"code_sha256": digest.hexdigest(), "git_commit": None, "git_dirty": None}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        if revision.returncode == 0:
            metadata["git_commit"] = revision.stdout.strip()
        if status.returncode == 0:
            metadata["git_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return metadata


def _environment_metadata() -> dict[str, object]:
    processor = platform.processor() or platform.machine()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        for line in cpu_info.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                processor = line.partition(":")[2].strip()
                break
    return {
        "python": platform.python_version(),
        "packages": {
            package: version(package) for package in ("numpy", "pandas", "scipy", "scikit-learn")
        },
        "platform": platform.platform(),
        "processor": processor,
        "logical_cpu_count": os.cpu_count(),
        "device": "cpu",
    }


def run_experiment(
    model_name: str,
    *,
    dataset_variant: str = "no_duplicates",
    config: ExperimentConfig | None = None,
    split_dir: str | Path = SPLITS_DIR,
    output_dir: str | Path | None = None,
) -> ExperimentResult:
    config = load_config() if config is None else config
    model = build_model(model_name, config)
    splits = load_splits(dataset_variant, split_dir=split_dir)
    threshold = None if model_name == "B0" else config.decision_threshold
    started_at = datetime.now(timezone.utc)
    run_id = f"{started_at:%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"

    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(splits.train["text"], splits.train["label"])
    train_seconds = perf_counter() - start

    start = perf_counter()
    predictions = predict_records(model, splits.validation, threshold=threshold)
    inference_seconds = perf_counter() - start
    metrics = evaluate_predictions(predictions)

    split_dir = Path(split_dir)
    settings = {
        "model": model_name,
        "dataset_variant": dataset_variant,
        "decision_rule": "most_frequent" if model_name == "B0" else "p_spam >= threshold",
        "decision_threshold": threshold,
        "shared_configuration": config.to_dict(),
        "preprocessing": preprocessing_config(model_name),
    }
    metadata = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "evaluation_split": "val",
        "split_manifest_sha256": file_sha256(split_dir / "manifest.json"),
        "train_sha256": file_sha256(split_dir / dataset_variant / "train.csv"),
        "validation_sha256": file_sha256(split_dir / dataset_variant / "val.csv"),
        "train_records": len(splits.train),
        "validation_records": len(splits.validation),
        "train_class_counts": splits.train["label"].value_counts().to_dict(),
        "validation_class_counts": splits.validation["label"].value_counts().to_dict(),
        "feature_count": (
            len(model.named_steps["tfidf"].vocabulary_) if model_name != "B0" else None
        ),
        "preprocessing_summary": {
            "train": preprocessing_summary(splits.train["text"], variant=model_name),
            "val": preprocessing_summary(splits.validation["text"], variant=model_name),
        },
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "training_scope": "fit the complete model pipeline on training text",
        "inference_scope": "validation text to probabilities, labels and prediction table",
        "api_cost": 0.0,
        "environment": _environment_metadata(),
        **_code_metadata(),
    }
    run_dir = (
        Path(output_dir) / dataset_variant / model_name / run_id
        if output_dir is not None else None
    )
    result = ExperimentResult(
        model_name, dataset_variant, model, predictions, metrics,
        train_seconds, inference_seconds, settings, metadata, run_dir,
    )
    if run_dir is not None:
        _save_result(result)
    return result


def _save_result(result: ExperimentResult) -> None:
    run_dir = result.output_dir
    if run_dir is None:
        raise ValueError("An output directory is required to save results.")
    run_dir.mkdir(parents=True, exist_ok=False)
    result.predictions.to_csv(run_dir / "predictions.csv", index=False, lineterminator="\n")
    for name, values in (
        ("config", result.config), ("metrics", result.metrics), ("metadata", result.metadata)
    ):
        (run_dir / f"{name}.json").write_text(
            json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(BASELINE_NAMES))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    results = [
        run_experiment(
            name, dataset_variant=dataset, config=config,
            split_dir=arguments.split_dir, output_dir=arguments.output_dir,
        )
        for dataset in arguments.datasets
        for name in arguments.models
    ]
    summary = pd.DataFrame([result.summary() for result in results])
    summary.to_csv(arguments.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
