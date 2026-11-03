"""Build classifiers with a shared TF-IDF and logistic regression configuration."""

from dataclasses import asdict

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sms_spam.config import ExperimentConfig
from sms_spam.preprocessing import PREPROCESSORS, transform_texts

BASELINE_NAMES = ("B0", "B1")
MODEL_NAMES = BASELINE_NAMES + tuple(PREPROCESSORS)
BaselineModel = DummyClassifier | Pipeline


def build_model(name: str, config: ExperimentConfig) -> BaselineModel:
    if name == "B0":
        return DummyClassifier(strategy="most_frequent")
    if name in MODEL_NAMES:
        steps = []
        if name in PREPROCESSORS:
            steps.append(("preprocessing", FunctionTransformer(
                transform_texts, kw_args={"variant": name}, validate=False,
            )))
        steps.extend([
            ("tfidf", TfidfVectorizer(**asdict(config.tfidf))),
            ("classifier", LogisticRegression(**asdict(config.logistic_regression))),
        ])
        return Pipeline(steps)
    raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}.")
