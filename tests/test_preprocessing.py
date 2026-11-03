import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from sms_spam.config import load_config
from sms_spam.data import PROJECT_ROOT, RECORD_COLUMNS, file_sha256
from sms_spam.experiments import run_experiment
from sms_spam.models import build_model
from sms_spam.preprocessing import (
    PHONE_TOKEN,
    PREPROCESSORS,
    URL_TOKEN,
    mask_phone_numbers,
    mask_urls,
    normalize_letter_repetitions,
    preprocessing_config,
    preprocessing_summary,
    transform_texts,
)


class TransformationTests(unittest.TestCase):
    def test_masks_supported_urls(self):
        for address in (
            "http://example.com", "https://example.com/prize?q=123#claim",
            "www.example.com", "EXAMPLE.CO.UK", "offers.example.org/path",
        ):
            with self.subTest(address=address):
                self.assertEqual(mask_urls(address).strip(), URL_TOKEN)

    def test_preserves_punctuation_quotes_and_text_after_urls(self):
        message = 'Visit "https://example.com/prize", then example.org. Thanks!'
        expected = f'Visit " {URL_TOKEN} ", then  {URL_TOKEN} . Thanks!'
        self.assertEqual(mask_urls(message), expected)
        message = "See https://example.com/path,please reply."
        self.assertEqual(mask_urls(message), f"See  {URL_TOKEN} ,please reply.")

    def test_does_not_mask_email_addresses_or_plain_text(self):
        for message in (
            "Email user@example.com or user@sub.example.com",
            "No links here. Call tomorrow!", "Hello e.g. friend", "example.company",
        ):
            with self.subTest(message=message):
                self.assertEqual(mask_urls(message), message)

    def test_masks_phones_and_shortcodes(self):
        for number in (
            "08001234567", "0871-872-9758", "01234 567890", "+44 7700 900123",
            "+447700900123", "(020) 7946 0958", "1234567890", "0044 7700 900123",
            "82277", "123456",
        ):
            with self.subTest(number=number):
                self.assertEqual(mask_phone_numbers(number).strip(), PHONE_TOKEN)

    def test_preserves_amounts_dates_times_urls_emails_and_other_numbers(self):
        for message in (
            "£10000", "10000 GBP", "GBP 10000", "50000 pounds", "$100000000",
            "10000€", "12345.67", "12345,678", "03/11/2024", "2024-11-03",
            "12:30:45", "2024", "1234", "1234567", "1234567890123456",
            "item12345", "12345abc", "12345@example.com", "user@12345.example.com",
            "https://example.com/12345", "www.example.com/01234567890",
        ):
            with self.subTest(message=message):
                self.assertEqual(mask_phone_numbers(message), message)

    def test_masks_multiple_contacts_without_touching_the_price(self):
        message = "Call 0871-872-9758 or text 82277. Prize: £10000."
        transformed = mask_phone_numbers(message)
        self.assertEqual(transformed.count(PHONE_TOKEN), 2)
        self.assertIn("Prize: £10000.", transformed)

    def test_normalizes_letters_without_changing_case_digits_or_punctuation(self):
        self.assertEqual(
            normalize_letter_repetitions("Soooo FREEEEE żżżż!!! 11111 ²²² ... AaAa"),
            "Soo FREE żż!!! 11111 ²²² ... AaAa",
        )
        self.assertEqual(
            normalize_letter_repetitions("Good book, see you!!"), "Good book, see you!!"
        )

    def test_transformations_are_idempotent_and_keep_unmatched_text_unchanged(self):
        examples = (
            "Call +44 7700 900123 or text 82277. Visit https://example.com/path!",
            "Soooo happy!!! 11111", "  Meet me tomorrow.  ",
        )
        for variant, transform in PREPROCESSORS.items():
            for text in examples:
                with self.subTest(variant=variant, text=text):
                    self.assertEqual(transform(transform(text)), transform(text))
            self.assertEqual(transform(examples[-1]), examples[-1])

    def test_change_counts_and_unknown_variants(self):
        messages = ["Visit example.com", "Hello friend"]
        self.assertEqual(preprocessing_summary(messages, variant="P1"), {
            "records": 2, "changed_records": 1, "fraction_changed": 0.5,
        })
        self.assertEqual(preprocessing_summary(messages, variant="B1")["changed_records"], 0)
        for function in (transform_texts, preprocessing_summary):
            with self.assertRaisesRegex(ValueError, "Unknown preprocessing"):
                function(messages, variant="unknown")
        with self.assertRaisesRegex(ValueError, "Unknown preprocessing"):
            preprocessing_config("unknown")

    def test_each_variant_changes_only_its_selected_operation(self):
        message = "Soooo visit example.com or call 08001234567!"
        p1, p2, p3 = [transform_texts([message], variant=variant)[0] for variant in PREPROCESSORS]
        self.assertIn("Soooo", p1)
        self.assertIn("08001234567", p1)
        self.assertIn("Soooo", p2)
        self.assertIn("example.com", p2)
        self.assertIn("example.com", p3)
        self.assertIn("08001234567", p3)


class PreprocessingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.train = pd.DataFrame([
            ("sms_000001", "ham", "Hello friend say helloooo today"),
            ("sms_000002", "ham", "Meet tomorrow at noon"),
            ("sms_000003", "spam", "Win visit https://offers.example.com or call 08001234567"),
            ("sms_000004", "spam", "Claim now visit www.example.org or text 82277"),
        ], columns=RECORD_COLUMNS)
        self.validation = pd.DataFrame([
            ("sms_000005", "ham", "Hello validationonly"),
            ("sms_000006", "spam", "Claim visit https://new.example.com text 83333 validationonly"),
        ], columns=RECORD_COLUMNS)

    def test_all_variants_keep_the_baseline_vectorizer_and_classifier_settings(self):
        baseline = build_model("B1", self.config)
        for variant in PREPROCESSORS:
            with self.subTest(variant=variant):
                model = build_model(variant, self.config)
                self.assertEqual([name for name, _ in model.steps], [
                    "preprocessing", "tfidf", "classifier",
                ])
                for step in ("tfidf", "classifier"):
                    self.assertEqual(
                        model.named_steps[step].get_params(),
                        baseline.named_steps[step].get_params(),
                    )

    def test_markers_are_single_features_and_vocabulary_is_fitted_on_training_only(self):
        for variant, expected_token in (("P1", URL_TOKEN), ("P2", PHONE_TOKEN), ("P3", "helloo")):
            with self.subTest(variant=variant):
                model = build_model(variant, self.config).fit(
                    self.train["text"], self.train["label"]
                )
                vectorizer = model.named_steps["tfidf"]
                self.assertIn(expected_token, vectorizer.vocabulary_)
                self.assertEqual(vectorizer.build_analyzer()(expected_token), [expected_token])
                self.assertNotIn("validationonly", vectorizer.vocabulary_)
                transformed = transform_texts(self.validation["text"], variant=variant)
                expected = model.named_steps["classifier"].predict_proba(
                    vectorizer.transform(transformed)
                )
                np.testing.assert_allclose(model.predict_proba(self.validation["text"]), expected)

    def test_runner_preserves_records_saves_rules_and_does_not_read_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_dir = root / "splits"
            manifest = {"schema_version": 1, "variants": {"no_duplicates": {}}}
            for name, frame in (("train", self.train), ("val", self.validation)):
                path = split_dir / "no_duplicates" / f"{name}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
                manifest["variants"]["no_duplicates"][name] = {
                    "sha256": file_sha256(path), "record_count": len(frame),
                    "class_counts": frame["label"].value_counts().to_dict(),
                }
            (split_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            before = {path: path.read_bytes() for path in split_dir.rglob("*") if path.is_file()}
            read_bytes = Path.read_bytes

            def guarded_read(path):
                if path.name == "test.csv":
                    raise AssertionError("Preprocessing experiment accessed test data")
                return read_bytes(path)

            with patch.object(Path, "read_bytes", guarded_read):
                for variant in PREPROCESSORS:
                    result = run_experiment(
                        variant, config=self.config, split_dir=split_dir,
                        output_dir=root / "results",
                    )
                    self.assertEqual(
                        result.predictions["record_id"].tolist(),
                        self.validation["record_id"].tolist(),
                    )
                    self.assertEqual(
                        result.predictions["y_true"].tolist(), self.validation["label"].tolist()
                    )
                    saved = json.loads((result.output_dir / "config.json").read_text())
                    self.assertEqual(saved["preprocessing"], preprocessing_config(variant))
                    self.assertEqual(saved["decision_threshold"], self.config.decision_threshold)
                    self.assertGreater(result.metadata["feature_count"], 0)
                    self.assertGreater(
                        result.metadata["preprocessing_summary"]["train"]["changed_records"], 0
                    )
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
            assert_frame_equal(self.train, pd.read_csv(split_dir / "no_duplicates/train.csv"))
            assert_frame_equal(self.validation, pd.read_csv(split_dir / "no_duplicates/val.csv"))

            process = subprocess.run([
                sys.executable, "-m", "sms_spam.experiments", "--datasets", "no_duplicates",
                "--models", "P1", "P2", "P3", "--split-dir", str(split_dir),
                "--output-dir", str(root / "cli_results"),
            ], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
            summary = pd.read_csv(root / "cli_results/summary.csv")
            self.assertEqual(summary["model"].tolist(), list(PREPROCESSORS))
            self.assertIn("f1_spam", process.stdout)


if __name__ == "__main__":
    unittest.main()
