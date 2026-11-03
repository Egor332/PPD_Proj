import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.exceptions import ConvergenceWarning

from sms_spam.config import ExperimentConfig, load_config
from sms_spam.data import PROJECT_ROOT, RECORD_COLUMNS, VARIANTS, file_sha256
from sms_spam.evaluation import evaluate_predictions, predict_records
from sms_spam.experiments import run_experiment
from sms_spam.models import build_model


class ConfigurationTests(unittest.TestCase):
    def test_checked_in_configuration_matches_defaults(self):
        self.assertEqual(load_config(), ExperimentConfig())

    def test_rejects_invalid_thresholds(self):
        for threshold in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                ExperimentConfig(decision_threshold=threshold)

    def test_rejects_unknown_configuration_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for settings in ({"unknown": 1}, {"tfidf": {"unknown": 1}}):
                with self.subTest(settings=settings):
                    path.write_text(json.dumps(settings), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.predictions = pd.DataFrame({
            "record_id": [f"sms_{index:06d}" for index in range(1, 5)],
            "y_true": ["ham", "ham", "spam", "spam"],
            "y_pred": ["ham", "spam", "ham", "spam"],
            "p_spam": [0.1, 0.8, 0.2, 0.9],
        })

    def test_metrics_and_confusion_matrix_use_spam_as_positive(self):
        metrics = evaluate_predictions(self.predictions)
        for name in ("f1_spam", "precision_spam", "recall_spam", "macro_f1", "accuracy"):
            self.assertEqual(metrics[name], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["class_labels"], ["ham", "spam"])

    def test_handles_no_positive_predictions_and_distinguishes_ap_from_pr_auc(self):
        predictions = self.predictions.assign(y_pred="ham", p_spam=0.0)
        metrics = evaluate_predictions(predictions)
        for name in ("f1_spam", "precision_spam", "recall_spam"):
            self.assertEqual(metrics[name], 0.0)
        self.assertEqual(metrics["average_precision"], 0.5)
        self.assertEqual(metrics["pr_auc"], 0.75)

    def test_rejects_invalid_predictions(self):
        cases = (
            self.predictions.drop(columns="p_spam"),
            self.predictions.assign(p_spam=float("nan")),
            self.predictions.assign(p_spam=1.1),
            self.predictions.assign(record_id="sms_000001"),
            self.predictions.assign(y_true="ham"),
            self.predictions.assign(y_pred="unknown"),
        )
        for predictions in cases:
            with self.subTest(columns=list(predictions.columns)), self.assertRaises(ValueError):
                evaluate_predictions(predictions)

    def test_probability_column_order_and_threshold_boundary(self):
        class ReversedClassesModel:
            classes_ = np.array(["spam", "ham"])

            def predict_proba(self, texts):
                return np.array([[0.5, 0.5], [0.49, 0.51]])

        records = pd.DataFrame({
            "record_id": ["sms_000001", "sms_000002"],
            "label": ["spam", "ham"],
            "text": ["First message", "Second message"],
        })
        predictions = predict_records(ReversedClassesModel(), records, threshold=0.5)
        self.assertEqual(predictions["y_pred"].tolist(), ["spam", "ham"])
        self.assertEqual(predictions["p_spam"].tolist(), [0.5, 0.49])


class BaselineExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.split_dir = self.root / "splits"
        self.output_dir = self.root / "results"
        self.frames = {
            "train": pd.DataFrame([
                ("sms_000001", "ham", "friendly greeting one"),
                ("sms_000002", "ham", "friendly greeting two"),
                ("sms_000003", "spam", "urgent prize one"),
                ("sms_000004", "spam", "urgent prize two"),
            ], columns=RECORD_COLUMNS),
            "val": pd.DataFrame([
                ("sms_000005", "ham", "friendly validationonly"),
                ("sms_000006", "spam", "urgent validationonly"),
            ], columns=RECORD_COLUMNS),
            "test": pd.DataFrame([
                ("sms_000007", "ham", "testonly greeting"),
                ("sms_000008", "spam", "testonly prize"),
            ], columns=RECORD_COLUMNS),
        }
        manifest = {"schema_version": 1, "variants": {}}
        for variant in VARIANTS:
            manifest["variants"][variant] = {}
            for name, frame in self.frames.items():
                path = self.split_dir / variant / f"{name}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
                manifest["variants"][variant][name] = {
                    "sha256": file_sha256(path),
                    "record_count": len(frame),
                    "class_counts": frame["label"].value_counts().to_dict(),
                }
        (self.split_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_b0_predicts_training_majority_independently_of_threshold(self):
        for variant in VARIANTS:
            path = self.split_dir / variant / "train.csv"
            train = self.frames["train"].copy()
            train.loc[2, "label"] = "ham"
            train.to_csv(path, index=False)
            manifest_path = self.split_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["variants"][variant]["train"].update(
                sha256=file_sha256(path), class_counts={"ham": 3, "spam": 1}
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = run_experiment(
                "B0", dataset_variant=variant,
                config=ExperimentConfig(decision_threshold=0.0), split_dir=self.split_dir,
            )
            self.assertEqual(result.predictions["y_pred"].tolist(), ["ham", "ham"])
            self.assertEqual(result.metrics["f1_spam"], 0.0)
            self.assertIsNone(result.config["decision_threshold"])

    def test_b1_learns_vocabulary_and_idf_only_from_training(self):
        result = run_experiment("B1", split_dir=self.split_dir)
        vectorizer = result.model.named_steps["tfidf"]
        self.assertNotIn("validationonly", vectorizer.vocabulary_)
        self.assertNotIn("testonly", vectorizer.vocabulary_)
        friendly_index = vectorizer.vocabulary_["friendly"]
        self.assertAlmostEqual(vectorizer.idf_[friendly_index], np.log(5 / 3) + 1)
        self.assertEqual(result.predictions["record_id"].tolist(), ["sms_000005", "sms_000006"])
        self.assertEqual(result.metrics["f1_spam"], 1.0)

    def test_experiment_does_not_read_test_data_or_modify_splits(self):
        before = {path: path.read_bytes() for path in self.split_dir.rglob("*") if path.is_file()}
        read_bytes = Path.read_bytes

        def guarded_read(path):
            if path.name == "test.csv":
                raise AssertionError("Experiment accessed test data")
            return read_bytes(path)

        with patch.object(Path, "read_bytes", guarded_read):
            run_experiment("B1", split_dir=self.split_dir, output_dir=self.output_dir)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_saves_recomputable_predictions_configuration_and_metadata(self):
        result = run_experiment("B1", split_dir=self.split_dir, output_dir=self.output_dir)
        predictions = pd.read_csv(result.output_dir / "predictions.csv")
        assert_frame_equal(predictions, result.predictions)
        metrics = json.loads((result.output_dir / "metrics.json").read_text())
        self.assertEqual(metrics, evaluate_predictions(predictions))
        config = json.loads((result.output_dir / "config.json").read_text())
        self.assertEqual(config["decision_threshold"], 0.5)
        self.assertEqual(config["shared_configuration"]["logistic_regression"]["solver"], "lbfgs")
        metadata = json.loads((result.output_dir / "metadata.json").read_text())
        self.assertEqual(metadata["evaluation_split"], "val")
        self.assertEqual(
            metadata["validation_sha256"], file_sha256(self.split_dir / "no_duplicates/val.csv")
        )
        self.assertIn("scikit-learn", metadata["environment"]["packages"])
        self.assertEqual(len(metadata["code_sha256"]), 64)
        self.assertGreaterEqual(metadata["train_seconds"], 0)
        self.assertGreaterEqual(metadata["inference_seconds"], 0)

    def test_repeated_runs_preserve_predictions_and_previous_artifacts(self):
        first = run_experiment("B1", split_dir=self.split_dir, output_dir=self.output_dir)
        contents = {path: path.read_bytes() for path in first.output_dir.iterdir()}
        second = run_experiment("B1", split_dir=self.split_dir, output_dir=self.output_dir)
        assert_frame_equal(first.predictions, second.predictions)
        self.assertNotEqual(first.output_dir, second.output_dir)
        for path, content in contents.items():
            self.assertEqual(path.read_bytes(), content)

    def test_nonconverged_fit_does_not_produce_results(self):
        def nonconverged_fit(*args):
            warnings.warn("Model did not converge", ConvergenceWarning)

        with patch("sms_spam.experiments.build_model") as factory:
            factory.return_value.fit.side_effect = nonconverged_fit
            with self.assertRaises(ConvergenceWarning):
                run_experiment("B1", split_dir=self.split_dir, output_dir=self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_rejects_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model("unknown", ExperimentConfig())

    def test_cli_runs_both_models_and_writes_summary(self):
        process = subprocess.run([
            sys.executable, "-m", "sms_spam.experiments", "--datasets", "no_duplicates",
            "--split-dir", str(self.split_dir), "--output-dir", str(self.output_dir),
        ], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        summary = pd.read_csv(self.output_dir / "summary.csv")
        self.assertEqual(summary["model"].tolist(), ["B0", "B1"])
        self.assertEqual(summary["dataset_variant"].tolist(), ["no_duplicates", "no_duplicates"])
        self.assertIn("f1_spam", process.stdout)


if __name__ == "__main__":
    unittest.main()
