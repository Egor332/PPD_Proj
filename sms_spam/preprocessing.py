"""Deterministic text transformations shared by training and inference."""

import re
from collections.abc import Iterable

URL_TOKEN = "zzurltokenzz"
PHONE_TOKEN = "zzphonetokenzz"

_URL_PATTERN = re.compile(
    r"(?<![\w@.-])(?:https?://[^\s<>\"',;]+|www\.[a-z0-9][^\s<>\"',;]*|"
    r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
    r"(?:com|net|org|uk|info|biz|tv|cc|us|eu|de|in|mobi|ws|name|io)"
    r"(?![\w-]|\.[a-z0-9])(?::[0-9]{1,5})?(?:[/?#][^\s<>\"',;]*)?)",
    re.IGNORECASE,
)
_PHONE_PATTERN = re.compile(
    r"(?<![\w.+-])(?:\+[1-9][0-9]{8,14}|"
    r"(?:\+[1-9][0-9]{0,2}[ -])?(?:\([0-9]{2,5}\)|[0-9]{2,5})"
    r"(?:[ -][0-9]{2,6}){1,3}|[0-9]{9,15}|[0-9]{5,6})(?!\w)"
)
_PROTECTED_PHONE_PATTERN = re.compile(
    rf"{_URL_PATTERN.pattern}|"
    r"[^\s@]+@[^\s@]+\.[^\s@]+|"
    r"(?:[£$€]\s*|\b(?:GBP|USD|EUR)\s*)[+-]?[0-9][0-9,]*(?:\.[0-9]+)?|"
    r"\b[0-9][0-9,]*(?:\.[0-9]+)?\s*"
    r"(?:(?:GBP|USD|EUR|pounds?|dollars?|euros?|pence|p)\b|[£$€])|"
    r"\b(?:[0-9]{1,2}[-/.][0-9]{1,2}[-/.][0-9]{2,4}|"
    r"[0-9]{4}[-/.][0-9]{1,2}[-/.][0-9]{1,2})\b|"
    r"\b[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?\b|"
    r"\b[0-9]+[.,][0-9]+\b",
    re.IGNORECASE,
)
_REPEATED_LETTER_PATTERN = re.compile(r"([^\W\d_])\1{2,}")


def mask_urls(text: str) -> str:
    def replace(match: re.Match) -> str:
        value = match.group()
        address = value.rstrip(".!?:)]}")
        return f" {URL_TOKEN} {value[len(address):]}"

    return _URL_PATTERN.sub(replace, text)


def mask_phone_numbers(text: str) -> str:
    protected_spans = [match.span() for match in _PROTECTED_PHONE_PATTERN.finditer(text)]

    def replace(match: re.Match) -> str:
        value = match.group()
        digits = sum(character.isdigit() for character in value)
        is_phone = 9 <= digits <= 15
        is_shortcode = value.isdigit() and len(value) in (5, 6)
        overlaps_protected = any(
            match.start() < end and match.end() > start for start, end in protected_spans
        )
        if (is_phone or is_shortcode) and not overlaps_protected:
            return f" {PHONE_TOKEN} "
        return value

    return _PHONE_PATTERN.sub(replace, text)


def normalize_letter_repetitions(text: str) -> str:
    def replace(match: re.Match) -> str:
        letter = match.group(1)
        return letter * 2 if letter.isalpha() else match.group()

    return _REPEATED_LETTER_PATTERN.sub(replace, text)


PREPROCESSORS = {
    "P1": mask_urls,
    "P2": mask_phone_numbers,
    "P3": normalize_letter_repetitions,
}


def transform_texts(texts: Iterable[str], *, variant: str) -> list[str]:
    if variant not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocessing variant {variant!r}.")
    transform = PREPROCESSORS[variant]
    return [transform(text) for text in texts]


def preprocessing_config(variant: str) -> dict[str, object]:
    if variant in ("B0", "B1"):
        return {"name": "identity"}
    if variant == "P1":
        return {
            "name": "mask_urls", "token": URL_TOKEN,
            "pattern": _URL_PATTERN.pattern, "ignore_case": True,
            "preserved_trailing_punctuation": ".!?:)]}",
        }
    if variant == "P2":
        return {
            "name": "mask_phone_numbers", "token": PHONE_TOKEN,
            "pattern": _PHONE_PATTERN.pattern,
            "protected_pattern": _PROTECTED_PHONE_PATTERN.pattern,
            "protected_ignore_case": True,
            "phone_digit_range": [9, 15], "shortcode_lengths": [5, 6],
        }
    if variant == "P3":
        return {
            "name": "normalize_letter_repetitions",
            "pattern": _REPEATED_LETTER_PATTERN.pattern, "replacement": r"\1\1",
            "letters_only": True, "letter_check": "str.isalpha", "case_sensitive": True,
        }
    raise ValueError(f"Unknown preprocessing variant {variant!r}.")


def preprocessing_summary(texts: Iterable[str], *, variant: str) -> dict[str, object]:
    messages = list(texts)
    transformed = (
        messages if variant in ("B0", "B1") else transform_texts(messages, variant=variant)
    )
    changed = sum(original != result for original, result in zip(messages, transformed))
    return {
        "records": len(messages),
        "changed_records": changed,
        "fraction_changed": changed / len(messages) if messages else 0.0,
    }
