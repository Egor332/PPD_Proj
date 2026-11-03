import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from create_splits import prepare_splits
from sms_spam.data import (
    PROJECT_ROOT,
    DataSplits,
    file_sha256,
    load_raw_data,
    load_splits,
    validate_records,
    validate_splits,
)


class RawDataTests(unittest.TestCase):
    def test_preserves_text_and_assigns_source_line_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            path.write_bytes(b'ham\t  NA\t"quoted" text  \r\nspam\tPromotion\n')
            records = load_raw_data(path)

        self.assertEqual(records["record_id"].tolist(), ["sms_000001", "sms_000002"])
        self.assertEqual(records["text"].tolist(), ['  NA\t"quoted" text  ', "Promotion"])

    def test_rejects_invalid_source_records(self):
        cases = (
            (b"ham without tab\n", "missing tab"),
            (b"other\tMessage\n", "labels must"),
            (b"ham\t  \n", "whitespace-only"),
            (b"ham\tSame\nspam\tSame\n", "conflicting labels"),
            (b"", "no records"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            for content, message in cases:
                with self.subTest(content=content):
                    path.write_bytes(content)
                    with self.assertRaisesRegex(ValueError, message):
                        load_raw_data(path)

            path.write_bytes(b"ham\t\xff\n")
            with self.assertRaises(UnicodeDecodeError):
                load_raw_data(path)

    def test_rejects_missing_values_and_invalid_ids(self):
        records = pd.DataFrame({"record_id": ["sms_000001"], "label": ["ham"], "text": ["Hello"]})
        for column, value in (("record_id", "invalid"), ("label", None), ("text", None)):
            with self.subTest(column=column):
                invalid = records.copy()
                invalid.loc[0, column] = value
                with self.assertRaises(ValueError):
                    validate_records(invalid)

        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_records(records.drop(columns="record_id"))


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        rows = [f"ham\tHam message {index // 2}\n" for index in range(60)]
        rows += [f"spam\tSpam promotion {index}\n" for index in range(20)]
        self.source.write_text("".join(rows), encoding="utf-8")
        self.output = self.root / "splits"
        self.prepared = prepare_splits(self.source, self.output)

    def test_keeps_milestone_one_assignments_and_order(self):
        expected_hashes = {
            "with_duplicates": {
                "train": "97ca1f5b6e9437689f2331bd0a71ea6e8e2a8eec64ae843ba3bd5976a01cfa42",
                "val": "32f6070edac791131e6773d4f1f2e7e508ba554173796ef052d81f91ac81def9",
                "test": "99df8194a4e4a3874584b72eed6ff11d86d007f87344d31307ff90bc7394e669",
            },
            "no_duplicates": {
                "train": "7e8e8e764dfa2db07fbc712909873e087366bad336e0d00667eec93738bdd0ee",
                "val": "adefc17d02940a6aff089cdeddb86b92aedb39aaa408cdb35dc48bf957422412",
                "test": "29f980d162d198f549980beb34cb39c4aed86d06d3a7fa673c5804d0574c2b2a",
            },
        }
        for variant, splits in self.prepared.items():
            for name, frame in splits.items():
                assignment_hash = hashlib.sha256("\n".join(frame["record_id"]).encode()).hexdigest()
                self.assertEqual(assignment_hash, expected_hashes[variant][name])

    def test_every_source_id_occurs_once_and_deduplication_keeps_first(self):
        records = load_raw_data(self.source)
        for variant, splits in self.prepared.items():
            combined = pd.concat([frame for _, frame in splits.items()], ignore_index=True)
            expected = records if variant == "with_duplicates" else records.drop_duplicates("text")
            self.assertEqual(set(combined["record_id"]), set(expected["record_id"]))
            self.assertTrue(combined["record_id"].is_unique)
            reference = records.set_index("record_id").loc[combined["record_id"]].reset_index()
            assert_frame_equal(combined, reference)

    def test_default_loader_does_not_open_test(self):
        with patch("sms_spam.data.pd.read_csv", wraps=pd.read_csv) as read_csv:
            splits = load_splits(split_dir=self.output)

        self.assertIsNone(splits.test)
        self.assertEqual(
            [call.args[0].name for call in read_csv.call_args_list], ["train.csv", "val.csv"]
        )

    def test_loads_both_variants_with_explicit_test_access(self):
        for variant, expected in self.prepared.items():
            actual = load_splits(variant, split_dir=self.output, include_test=True)
            for (_, actual_frame), (_, expected_frame) in zip(actual.items(), expected.items()):
                assert_frame_equal(actual_frame, expected_frame)

    def test_repeat_preparation_is_byte_identical(self):
        before = {
            path.relative_to(self.output): path.read_bytes()
            for path in self.output.rglob("*") if path.is_file()
        }
        prepare_splits(self.source, self.output)
        second_output = self.root / "second_splits"
        prepare_splits(self.source, second_output)
        for relative_path, content in before.items():
            self.assertEqual((self.output / relative_path).read_bytes(), content)
            self.assertEqual((second_output / relative_path).read_bytes(), content)

    def test_upgrades_legacy_csv_without_changing_records(self):
        for path in self.output.rglob("*.csv"):
            records = pd.read_csv(path, dtype=str, keep_default_na=False)
            records[["label", "text"]].to_csv(path, index=False)
        prepare_splits(self.source, self.output)
        for variant, expected in self.prepared.items():
            actual = load_splits(variant, split_dir=self.output, include_test=True)
            for (_, actual_frame), (_, expected_frame) in zip(actual.items(), expected.items()):
                assert_frame_equal(actual_frame, expected_frame)

    def test_refuses_to_overwrite_changed_assignments_before_writing_any_files(self):
        path = self.output / "no_duplicates" / "test.csv"
        records = pd.read_csv(path, dtype=str, keep_default_na=False)
        records.iloc[::-1].to_csv(path, index=False)
        before = {path: path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "refusing to change"):
            prepare_splits(self.source, self.output)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_refuses_changed_source_version(self):
        self.source.write_text(self.source.read_text() + "ham\tNew message\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Source data changed"):
            prepare_splits(self.source, self.output)

    def test_loader_detects_modified_file(self):
        path = self.output / "no_duplicates" / "val.csv"
        with path.open("a", encoding="utf-8") as output:
            output.write("sms_999999,ham,Unexpected message\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_splits(split_dir=self.output)

    def test_manifest_records_source_version_and_seed(self):
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["dataset"]["sha256"], file_sha256(self.source))
        self.assertEqual(manifest["dataset"]["record_count"], 80)
        self.assertEqual(manifest["split"]["seed"], 42)

    def test_rejects_record_id_overlap_and_unexpected_text_duplicates(self):
        splits = self.prepared["no_duplicates"]
        invalid_validation = splits.validation.copy()
        invalid_validation.loc[0, "record_id"] = splits.train.iloc[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "duplicate record IDs"):
            validate_splits(DataSplits(splits.train, invalid_validation), variant="no_duplicates")

        invalid_validation = splits.validation.copy()
        invalid_validation.loc[0, ["label", "text"]] = (
            splits.train.iloc[0][["label", "text"]].to_numpy()
        )
        with self.assertRaisesRegex(ValueError, "identical messages"):
            validate_splits(DataSplits(splits.train, invalid_validation), variant="no_duplicates")
        validate_splits(DataSplits(splits.train, invalid_validation), variant="with_duplicates")

    def test_rejects_unknown_variant_and_missing_manifest(self):
        with self.assertRaisesRegex(ValueError, "Unknown variant"):
            load_splits("unknown", split_dir=self.output)
        with self.assertRaisesRegex(FileNotFoundError, "manifest"):
            load_splits(split_dir=self.root / "missing")

    def test_importing_preparation_script_has_no_side_effects(self):
        code = f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); import create_splits"
        before = set(self.root.iterdir())
        process = subprocess.run(
            [sys.executable, "-c", code], cwd=self.root, capture_output=True, text=True, check=True
        )
        self.assertEqual(process.stdout, "")
        self.assertEqual(set(self.root.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
