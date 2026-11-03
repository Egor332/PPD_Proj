import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from sms_spam.audit import audit_duplicates, normalize_for_audit
from sms_spam.data import DataSplits, RECORD_COLUMNS


def make_records(rows):
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)


class DuplicateAuditTests(unittest.TestCase):
    def setUp(self):
        prefix = "Your parcel is ready for collection at the local post office. Reference "
        self.splits = DataSplits(
            train=make_records([
                ("sms_000001", "ham", "Hello world"),
                ("sms_000002", "ham", "Hello world"),
                ("sms_000003", "spam", prefix + "12345"),
            ]),
            validation=make_records([
                ("sms_000004", "ham", "Hello world"),
                ("sms_000005", "ham", "  HELLO   ＷＯＲＬＤ  "),
                ("sms_000006", "spam", prefix + "12346"),
            ]),
            test=make_records([
                ("sms_000007", "ham", "Hello world"),
                ("sms_000008", "spam", "Different promotion"),
            ]),
        )

    def test_normalization_handles_unicode_case_and_whitespace(self):
        self.assertEqual(normalize_for_audit("  HELLO\t ＷＯＲＬＤ\n"), "hello world")

    def test_counts_duplicate_rows_and_all_split_pairs(self):
        audit = audit_duplicates(self.splits)
        within = audit.within_splits.set_index("split")
        self.assertEqual(within.loc["train", "exact_duplicate_rows"], 1)
        self.assertEqual(within.loc["val", "exact_duplicate_rows"], 0)
        self.assertEqual(within.loc["val", "normalized_duplicate_rows"], 1)
        between = audit.between_splits.set_index(["left_split", "right_split"])
        self.assertEqual(set(between.index), {("train", "val"), ("train", "test"), ("val", "test")})
        self.assertEqual(between.loc[("train", "val"), "shared_exact_texts"], 1)
        self.assertEqual(between.loc[("train", "val"), "left_exact_overlap_rows"], 2)
        self.assertEqual(between.loc[("train", "val"), "right_exact_overlap_rows"], 1)
        self.assertEqual(between.loc[("train", "val"), "shared_normalized_texts"], 1)

    def test_reports_similar_messages_as_candidates(self):
        audit = audit_duplicates(self.splits)
        self.assertEqual(audit.near_duplicates["left_id"].tolist(), ["sms_000003"])
        self.assertEqual(audit.near_duplicates["right_id"].tolist(), ["sms_000006"])
        self.assertGreaterEqual(audit.near_duplicates.iloc[0]["similarity"], 0.90)

    def test_keeps_inputs_unchanged_and_results_independent_of_batch_size(self):
        before = {name: frame.copy(deep=True) for name, frame in self.splits.items()}
        batched = audit_duplicates(self.splits, batch_size=1)
        unbatched = audit_duplicates(self.splits, batch_size=256)
        assert_frame_equal(batched.near_duplicates, unbatched.near_duplicates)
        for name, frame in self.splits.items():
            assert_frame_equal(frame, before[name])

    def test_excludes_normalized_equality_from_similarity_candidates(self):
        text = "A sufficiently long message repeated across different partitions."
        splits = DataSplits(
            make_records([("sms_000001", "ham", text)]),
            make_records([("sms_000002", "ham", text.upper())]),
        )
        audit = audit_duplicates(splits)
        self.assertEqual(audit.between_splits.iloc[0]["shared_normalized_texts"], 1)
        self.assertTrue(audit.near_duplicates.empty)

    def test_handles_short_messages_without_fitting_a_vectorizer(self):
        splits = DataSplits(
            make_records([("sms_000001", "ham", "Hi")]),
            make_records([("sms_000002", "ham", "Hello")]),
        )
        self.assertTrue(audit_duplicates(splits).near_duplicates.empty)

    def test_rejects_invalid_settings(self):
        for settings in (
            {"similarity_threshold": 0}, {"similarity_threshold": 1.1},
            {"min_length_ratio": 0}, {"min_length_ratio": 1.1},
            {"min_length": 2}, {"batch_size": 0},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                audit_duplicates(self.splits, **settings)


if __name__ == "__main__":
    unittest.main()
