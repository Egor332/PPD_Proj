"""Build unfitted baseline classifiers."""

from dataclasses import asdict

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sms_spam.config import ExperimentConfig

MODEL_NAMES = ("B0", "B1")
BaselineModel = DummyClassifier | Pipeline


def build_model(name: str, config: ExperimentConfig) -> BaselineModel:
    if name == "B0":
        return DummyClassifier(strategy="most_frequent")
    if name == "B1":
        return Pipeline([
            ("tfidf", TfidfVectorizer(**asdict(config.tfidf))),
            ("classifier", LogisticRegression(**asdict(config.logistic_regression))),
        ])
    raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}.")
