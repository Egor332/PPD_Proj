"""Shared configuration for the baseline and preprocessing experiments."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sms_spam.data import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "baselines.json"


@dataclass(frozen=True)
class TfidfConfig:
    analyzer: str = "word"
    ngram_range: tuple[int, int] = (1, 1)
    lowercase: bool = True
    strip_accents: str | None = None
    preprocessor: None = None
    tokenizer: None = None
    token_pattern: str = r"(?u)\b\w\w+\b"
    stop_words: str | None = None
    min_df: int | float = 1
    max_df: int | float = 1.0
    max_features: int | None = None
    binary: bool = False
    norm: str = "l2"
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = False


@dataclass(frozen=True)
class LogisticRegressionConfig:
    solver: str = "lbfgs"
    C: float = 1.0
    l1_ratio: float = 0.0
    class_weight: str | None = None
    fit_intercept: bool = True
    max_iter: int = 2000
    tol: float = 1e-4
    random_state: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    decision_threshold: float = 0.5
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    logistic_regression: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)

    def __post_init__(self) -> None:
        if (
            isinstance(self.decision_threshold, bool)
            or not isinstance(self.decision_threshold, (int, float))
            or not math.isfinite(self.decision_threshold)
            or not 0 <= self.decision_threshold <= 1
        ):
            raise ValueError("decision_threshold must be a finite number in [0, 1].")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"decision_threshold", "tfidf", "logistic_regression"}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError(
            "Configuration must contain only decision_threshold, tfidf and logistic_regression."
        )

    try:
        tfidf = dict(settings.get("tfidf", {}))
        if "ngram_range" in tfidf:
            tfidf["ngram_range"] = tuple(tfidf["ngram_range"])
        return ExperimentConfig(
            decision_threshold=settings.get("decision_threshold", 0.5),
            tfidf=TfidfConfig(**tfidf),
            logistic_regression=LogisticRegressionConfig(**settings.get("logistic_regression", {})),
        )
    except TypeError as error:
        raise ValueError(f"Invalid experiment configuration: {error}") from error
