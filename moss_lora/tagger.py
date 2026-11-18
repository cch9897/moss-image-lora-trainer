"""Booru tagger post-processing, vocabulary mapping and threshold calibration.

The tagger is optional: nothing here imports ``transformers``/``torch``/
``onnxruntime`` until a model is actually loaded, and loading is always
``local_files_only`` by default (the pipeline must never download weights).
Tests exercise the deterministic post-processing/calibration paths with mock
predictions and a fake ONNX session.

The tagger never writes ``artist``; artist resolution belongs to staging.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, fields
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageOps

from moss_lora.annotations import (
    NEGATIVE_TAG_KEYS,
    normalize_name,
    normalize_tag_key,
)

CATEGORIES = ("character", "general", "rating", "copyright", "artist", "meta")
RATING_VALUES = ("general", "sensitive", "questionable", "explicit")
# Structural/quantity tags are never character identities.
NON_CHARACTER_TAGS = frozenset((
    "1girl", "1boy", "2girls", "2boys", "3girls", "3boys", "multiple girls",
    "multiple boys", "multiple others", "solo", "solo focus", "male focus",
    "female", "male", "no humans", "group", "couple", "duo", "trio",
    "ass", "breasts", "large breasts", "small breasts",
))
CATEGORY_ALIASES = {
    "0": "general", "general": "general", "tag": "general", "tags": "general",
    "1": "artist", "artist": "artist", "artists": "artist",
    "3": "copyright", "copyright": "copyright", "series": "copyright",
    "4": "character", "character": "character", "characters": "character",
    "5": "meta", "meta": "meta", "metadata": "meta",
    "9": "rating", "rating": "rating", "ratings": "rating",
}
# Runtime metadata may live next to the thresholds in tagger_config.json. It is
# deliberately ignored by TaggerConfig.load so a calibrated file is reusable.
CONFIG_METADATA_KEYS = frozenset((
    "backend", "model", "tags_csv", "preprocess", "device", "providers", "top_k",
    "model_sha256", "tags_sha256", "model_bytes", "tags_bytes", "label_count",
    "input_size", "input_layout", "runtime", "calibration", "calibration_report",
    "generated_at", "report",
))


class TaggerError(RuntimeError):
    """Tagger model unavailable or predictions malformed."""


@dataclass(frozen=True)
class TagPrediction:
    tag: str
    score: float
    category: str = "general"


@dataclass
class TaggerConfig:
    character_threshold: float = 0.55
    general_threshold: float = 0.35
    rating_threshold: float = 0.50
    max_characters: int = 4
    max_tags: int = 40
    min_general_tags: int = 5

    def __post_init__(self) -> None:
        for name in ("character_threshold", "general_threshold", "rating_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.max_characters < 0 or self.max_tags < 0:
            raise ValueError("max_characters/max_tags must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TaggerConfig:
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown tagger config keys: {unknown}")
        return cls(**{key: value for key, value in values.items() if key in allowed})

    @classmethod
    def load(cls, path: str | Path) -> TaggerConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Tagger config must be a JSON object: {path}")
        values = {key: value for key, value in payload.items()
                  if key not in CONFIG_METADATA_KEYS}
        return cls.from_dict(values)


@dataclass(frozen=True)
class TagResult:
    character: tuple[str, ...]
    tags: tuple[str, ...]
    tag_confidence: dict[str, float]
    rating: str | None
    new_character: tuple[str, ...] = ()
    tags_low_confidence: bool = False
    character_missing: bool = True

    def row_fields(self) -> dict[str, Any]:
        return {
            "character": list(self.character),
            "character_missing": self.character_missing,
            "tags": list(self.tags),
            "tag_confidence": dict(self.tag_confidence),
            "rating": self.rating,
            "new_character": list(self.new_character),
            "tags_low_confidence": self.tags_low_confidence,
        }


# --------------------------------------------------------------------------- vocabularies


def build_vocabulary(values: Iterable[str]) -> dict[str, str]:
    """normalized key -> display form; first spelling wins deterministically."""
    vocab: dict[str, str] = {}
    for value in values:
        clean = normalize_name(value)
        key = normalize_tag_key(clean)
        if key and key not in vocab:
            vocab[key] = clean
    return vocab


def _vocab_lookup(value: str, vocab: Mapping[str, str]) -> tuple[str, bool]:
    key = normalize_tag_key(value)
    if key in vocab:
        return vocab[key], True
    return normalize_name(value), False


def normalize_category(value: str) -> str:
    return CATEGORY_ALIASES.get(str(value).strip().lower(), "general")


def read_tag_labels(
    path: str | Path,
    *,
    category_map: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Read a WD14/PixAI ``selected_tags.csv`` into ``(name, category)`` rows.

    WD14 v3 uses numeric categories (0 general, 4 character, 9 rating); PixAI
    checkpoints use the same numeric ids. Row order is preserved because it is
    the model's output order. The CSV is metadata only: no download is attempted.
    """
    import csv

    csv_path = Path(path)
    if not csv_path.is_file():
        raise TaggerError(f"Tag label CSV not found: {csv_path} (models are never downloaded)")
    overrides = {normalize_tag_key(key): normalize_category(value)
                 for key, value in (category_map or {}).items()}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(name) for name in (reader.fieldnames or [])]
        lowered = {name.lower(): name for name in fieldnames}
        name_key = next((lowered[key] for key in ("name", "tag", "label")
                         if key in lowered), None)
        category_key = next((lowered[key] for key in ("category", "type", "tag_type")
                             if key in lowered), None)
        if name_key is None:
            raise TaggerError(f"Tag label CSV has no name/tag column: {csv_path} ({fieldnames})")
        labels: list[tuple[str, str]] = []
        for row in reader:
            name = normalize_name(row.get(name_key) or "")
            if not name:
                continue
            category = (normalize_category(row.get(category_key) or "general")
                        if category_key else "general")
            category = overrides.get(normalize_tag_key(name), category)
            labels.append((name, category))
    if not labels:
        raise TaggerError(f"Tag label CSV contains no labels: {csv_path}")
    return labels


def _normalize_predictions(predictions: Iterable[TagPrediction]) -> list[TagPrediction]:
    result: list[TagPrediction] = []
    for prediction in predictions:
        tag = normalize_name(prediction.tag)
        if not tag:
            continue
        score = float(prediction.score)
        if not math.isfinite(score):
            raise TaggerError(f"non-finite tag score for {tag!r}")
        result.append(TagPrediction(tag=tag, score=score,
                                    category=normalize_category(prediction.category)))
    return result


# --------------------------------------------------------------------------- post-processing


def postprocess_tags(
    predictions: Iterable[TagPrediction],
    config: TaggerConfig | None = None,
    *,
    character_vocab: Mapping[str, str] | Iterable[str] | None = None,
    general_vocab: Mapping[str, str] | Iterable[str] | None = None,
) -> TagResult:
    """Apply mandatory thresholds, vocabulary mapping, tag caps and negatives."""
    config = config or TaggerConfig()
    if character_vocab is not None and not isinstance(character_vocab, Mapping):
        character_vocab = build_vocabulary(character_vocab)
    if general_vocab is not None and not isinstance(general_vocab, Mapping):
        general_vocab = build_vocabulary(general_vocab)
    character_vocab = ({normalize_tag_key(key): str(value)
                        for key, value in (character_vocab or {}).items()})
    general_vocab = ({normalize_tag_key(key): str(value)
                      for key, value in (general_vocab or {}).items()})

    normalized = _normalize_predictions(predictions)
    character_candidates: list[TagPrediction] = []
    general_candidates: list[TagPrediction] = []
    rating_scores: dict[str, float] = {}

    for prediction in normalized:
        key = normalize_tag_key(prediction.tag)
        if prediction.category == "rating":
            rating = key
            if rating in RATING_VALUES:
                rating_scores[rating] = max(rating_scores.get(rating, 0.0), prediction.score)
            continue
        if prediction.category == "character":
            if key not in NON_CHARACTER_TAGS and prediction.score >= config.character_threshold:
                character_candidates.append(prediction)
        elif prediction.category == "general":
            if key not in NEGATIVE_TAG_KEYS and prediction.score >= config.general_threshold:
                general_candidates.append(prediction)

    character_candidates.sort(key=lambda item: (-item.score, normalize_tag_key(item.tag)))
    characters: list[str] = []
    new_character: list[str] = []
    seen_characters: set[str] = set()
    for prediction in character_candidates:
        if len(characters) >= config.max_characters:
            break
        key = normalize_tag_key(prediction.tag)
        if key in seen_characters:
            continue
        seen_characters.add(key)
        display, matched = _vocab_lookup(prediction.tag, character_vocab)
        if not matched:
            new_character.append(display)
        characters.append(display)

    general_scores: dict[str, float] = {}
    general_display: dict[str, str] = {}
    for prediction in general_candidates:
        key = normalize_tag_key(prediction.tag)
        display, _ = _vocab_lookup(prediction.tag, general_vocab)
        display_key = normalize_tag_key(display)
        if display_key in NEGATIVE_TAG_KEYS:
            continue
        if prediction.score > general_scores.get(display_key, -1.0):
            general_scores[display_key] = prediction.score
            general_display[display_key] = display

    union_scores = dict(general_scores)
    union_display = dict(general_display)
    for prediction in character_candidates:
        key = normalize_tag_key(prediction.tag)
        display, _ = _vocab_lookup(prediction.tag, character_vocab)
        display_key = normalize_tag_key(display)
        if display_key and prediction.score > union_scores.get(display_key, -1.0):
            union_scores[display_key] = prediction.score
            union_display[display_key] = display

    ordered = sorted(union_scores.items(), key=lambda item: (-item[1], item[0]))
    selected = ordered[: config.max_tags]
    tags = tuple(union_display[key] for key, _ in selected)
    tag_confidence = {union_display[key]: round(score, 4) for key, score in selected}

    rating = None
    if rating_scores:
        candidate, score = max(rating_scores.items(), key=lambda item: (item[1], item[0]))
        if score >= config.rating_threshold:
            rating = candidate

    return TagResult(
        character=tuple(characters),
        tags=tags,
        tag_confidence=tag_confidence,
        rating=rating,
        new_character=tuple(new_character),
        tags_low_confidence=len(general_scores) < config.min_general_tags,
        character_missing=not characters,
    )


# --------------------------------------------------------------------------- calibration


def _truth_keys(values: Iterable[str], *, exclude_non_character: bool = False) -> set[str]:
    keys = {key for key in (normalize_tag_key(value) for value in values) if key}
    keys -= NEGATIVE_TAG_KEYS
    if exclude_non_character:
        keys -= NON_CHARACTER_TAGS
    return keys


def _threshold_metrics(
    predictions: Sequence[TagPrediction],
    truth: set[str],
    threshold: float,
    *,
    max_items: int,
) -> tuple[int, int, int]:
    selected = [item for item in predictions if item.score >= threshold]
    selected.sort(key=lambda item: (-item.score, normalize_tag_key(item.tag)))
    predicted = {normalize_tag_key(item.tag) for item in selected[:max_items]}
    true_positive = len(predicted & truth)
    return true_positive, len(predicted) - true_positive, len(truth) - true_positive


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def calibrate_character(
    predictions_per_image: Sequence[Sequence[TagPrediction]],
    truth_per_image: Sequence[Iterable[str]],
    *,
    thresholds: Sequence[float] | None = None,
    min_precision: float = 0.90,
    min_recall: float = 0.80,
    max_characters: int = 4,
    preferred_threshold: float | None = None,
    tie_tolerance: float = 0.001,
) -> dict[str, Any]:
    """Sweep character thresholds and choose the best F1 that meets both gates.

    When F1 is effectively tied (within ``tie_tolerance``), the threshold
    closest to ``preferred_threshold`` wins, so a 200-image sample cannot move
    the production default on noise alone.
    """
    if len(predictions_per_image) != len(truth_per_image):
        raise ValueError("predictions_per_image and truth_per_image must have the same length")
    thresholds = list(thresholds or [round(0.30 + 0.05 * index, 2) for index in range(12)])
    truths = [_truth_keys(values, exclude_non_character=True) for values in truth_per_image]
    normalized_images = [_normalize_predictions(item) for item in predictions_per_image]
    evaluated: list[dict[str, Any]] = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for predictions, truth in zip(normalized_images, truths):
            characters = [item for item in predictions
                          if item.category == "character"
                          and normalize_tag_key(item.tag) not in NON_CHARACTER_TAGS]
            image_tp, image_fp, image_fn = _threshold_metrics(
                characters, truth, threshold, max_items=max_characters)
            tp += image_tp
            fp += image_fp
            fn += image_fn
        metrics = _prf(tp, fp, fn)
        evaluated.append({"threshold": threshold, **metrics})
    eligible = [row for row in evaluated
                if row["precision"] >= min_precision and row["recall"] >= min_recall]
    pool = eligible or evaluated
    best_f1 = max(row["f1"] for row in pool)
    near = [row for row in pool if best_f1 - row["f1"] <= tie_tolerance]
    if preferred_threshold is None:
        chosen = max(near, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    else:
        chosen = min(near, key=lambda row: (
            abs(row["threshold"] - preferred_threshold), -row["f1"], -row["precision"]))
    return {
        "threshold": chosen["threshold"],
        "precision": round(chosen["precision"], 4),
        "recall": round(chosen["recall"], 4),
        "f1": round(chosen["f1"], 4),
        "meets_gates": bool(eligible),
        "min_precision": min_precision,
        "min_recall": min_recall,
        "preferred_threshold": preferred_threshold,
        "tie_tolerance": tie_tolerance,
        "evaluated": [{key: (round(value, 4) if isinstance(value, float) else value)
                       for key, value in row.items()} for row in evaluated],
    }


def calibrate_general(
    predictions_per_image: Sequence[Sequence[TagPrediction]],
    truth_per_image: Sequence[Iterable[str]],
    *,
    thresholds: Sequence[float] | None = None,
    max_tags: int = 40,
    exclude_per_image: Sequence[Iterable[str]] | None = None,
    exclude_labels: Iterable[str] | None = None,
    preferred_threshold: float | None = None,
    tie_tolerance: float = 0.001,
) -> dict[str, Any]:
    """Sweep general thresholds and choose maximum tag F1 against known booru tags.

    ``exclude_per_image`` (typically the row's characters/artists) and
    ``exclude_labels`` (the checkpoint's character labels) are removed from the
    truth before scoring, so character identities are not counted as missed
    general tags.
    """
    if len(predictions_per_image) != len(truth_per_image):
        raise ValueError("predictions_per_image and truth_per_image must have the same length")
    if exclude_per_image is not None and len(exclude_per_image) != len(truth_per_image):
        raise ValueError("exclude_per_image and truth_per_image must have the same length")
    thresholds = list(thresholds or [round(0.15 + 0.05 * index, 2) for index in range(15)])
    shared_exclude = _truth_keys(exclude_labels or [])
    truths: list[set[str]] = []
    for index, values in enumerate(truth_per_image):
        truth = _truth_keys(values)
        truth -= shared_exclude
        truth -= set(RATING_VALUES)
        if exclude_per_image is not None:
            truth -= _truth_keys(exclude_per_image[index])
        truths.append(truth)
    normalized_images = [_normalize_predictions(item) for item in predictions_per_image]
    evaluated: list[dict[str, Any]] = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for predictions, truth in zip(normalized_images, truths):
            general = [item for item in predictions
                       if item.category == "general"
                       and normalize_tag_key(item.tag) not in NEGATIVE_TAG_KEYS]
            image_tp, image_fp, image_fn = _threshold_metrics(
                general, truth, threshold, max_items=max_tags)
            tp += image_tp
            fp += image_fp
            fn += image_fn
        metrics = _prf(tp, fp, fn)
        evaluated.append({"threshold": threshold, **metrics})
    best_f1 = max(row["f1"] for row in evaluated)
    near = [row for row in evaluated if best_f1 - row["f1"] <= tie_tolerance]
    if preferred_threshold is None:
        chosen = max(near, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    else:
        chosen = min(near, key=lambda row: (
            abs(row["threshold"] - preferred_threshold), -row["f1"], -row["precision"]))
    return {
        "threshold": chosen["threshold"],
        "precision": round(chosen["precision"], 4),
        "recall": round(chosen["recall"], 4),
        "f1": round(chosen["f1"], 4),
        "preferred_threshold": preferred_threshold,
        "tie_tolerance": tie_tolerance,
        "evaluated": [{key: (round(value, 4) if isinstance(value, float) else value)
                       for key, value in row.items()} for row in evaluated],
    }


def _as_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return []
    try:
        return list(value)
    except TypeError:
        return [value]


def _character_truth(
    row: Mapping[str, Any],
    character_label_keys: set[str],
) -> set[str]:
    """Ground-truth character identities for one row.

    The metadata ``character`` field is authoritative but occasionally empty
    while its booru ``tags`` still carry the identity. A tag is counted as a
    character only when the checkpoint itself labels it as one; unknown names
    from the ``character`` field are always retained.
    """
    truth = _truth_keys(
        _as_values(row.get("character")) + _as_values(row.get("characters")),
        exclude_non_character=True,
    )
    if character_label_keys:
        truth |= _truth_keys(_as_values(row.get("tags"))) & character_label_keys
    return truth


def _general_truth(
    row: Mapping[str, Any],
    character_truth: set[str],
    character_label_keys: set[str],
) -> set[str]:
    truth = _truth_keys(_as_values(row.get("tags")))
    truth -= character_truth
    truth -= character_label_keys
    truth -= set(RATING_VALUES)
    truth -= _truth_keys(_as_values(row.get("artist")) + _as_values(row.get("artists")))
    return truth


def _score_summary(values: Sequence[float]) -> dict[str, Any]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return {"count": 0}
    last = len(clean) - 1

    def percentile(ratio: float) -> float:
        return clean[min(last, max(0, int(round(ratio * last))))]

    return {
        "count": len(clean),
        "min": round(clean[0], 4),
        "median": round(percentile(0.50), 4),
        "p90": round(percentile(0.90), 4),
        "max": round(clean[-1], 4),
    }


def evaluate_tag_results(
    predictions_per_image: Sequence[Sequence[TagPrediction]],
    truth_rows: Sequence[Mapping[str, Any]],
    config: TaggerConfig | None = None,
    *,
    character_vocab: Mapping[str, str] | Iterable[str] | None = None,
    general_vocab: Mapping[str, str] | Iterable[str] | None = None,
    character_labels: Iterable[str] | None = None,
    max_failures: int = 200,
) -> dict[str, Any]:
    """Evaluate the full post-processing path against known booru rows.

    Character metrics compare identities; general metrics compare the final
    ``tags`` after removing predicted characters and the row's characters,
    artists and the checkpoint's character vocabulary from the truth. Failures
    contain tag names only (never captions) and are capped by ``max_failures``.
    """
    if len(predictions_per_image) != len(truth_rows):
        raise ValueError("predictions_per_image and truth_rows must have the same length")
    config = config or TaggerConfig()
    if character_vocab is not None and not isinstance(character_vocab, Mapping):
        character_vocab = build_vocabulary(character_vocab)
    if general_vocab is not None and not isinstance(general_vocab, Mapping):
        general_vocab = build_vocabulary(general_vocab)
    character_label_keys = _truth_keys(character_labels or [])

    char_tp = char_fp = char_fn = 0
    gen_tp = gen_fp = gen_fn = 0
    char_images = char_hits = gen_images = gen_hits = 0
    low_confidence = character_missing = character_missing_with_truth = 0
    tags_per_image: list[int] = []
    ratings: Counter[str] = Counter()
    rating_truth_total = rating_correct = 0
    rating_confusion: Counter[str] = Counter()
    missed_scores: list[float] = []
    failures: list[dict[str, Any]] = []
    failure_count = 0

    for predictions, row in zip(predictions_per_image, truth_rows):
        normalized = _normalize_predictions(predictions)
        result = postprocess_tags(
            normalized, config, character_vocab=character_vocab, general_vocab=general_vocab)
        truth_char = _character_truth(row, character_label_keys)
        truth_gen = _general_truth(row, truth_char, character_label_keys)
        pred_char = {normalize_tag_key(tag) for tag in result.character}
        pred_gen = ({normalize_tag_key(tag) for tag in result.tags}
                    - pred_char - character_label_keys)

        image_char_tp = len(pred_char & truth_char)
        image_gen_tp = len(pred_gen & truth_gen)
        char_tp += image_char_tp
        char_fp += len(pred_char) - image_char_tp
        char_fn += len(truth_char) - image_char_tp
        gen_tp += image_gen_tp
        gen_fp += len(pred_gen) - image_gen_tp
        gen_fn += len(truth_gen) - image_gen_tp
        if truth_char:
            char_images += 1
            char_hits += int(bool(pred_char & truth_char))
        if truth_gen:
            gen_images += 1
            gen_hits += int(bool(pred_gen & truth_gen))
        low_confidence += int(result.tags_low_confidence)
        character_missing += int(result.character_missing)
        character_missing_with_truth += int(result.character_missing and bool(truth_char))
        tags_per_image.append(len(result.tags))
        if result.rating:
            ratings[result.rating] += 1
        truth_rating = _truth_keys(_as_values(row.get("tags"))) & set(RATING_VALUES)
        if len(truth_rating) == 1:
            expected_rating = next(iter(truth_rating))
            rating_truth_total += 1
            rating_correct += int(result.rating == expected_rating)
            rating_confusion[f"{expected_rating}->{result.rating or 'none'}"] += 1

        char_candidates = [
            item for item in normalized
            if item.category == "character"
            and normalize_tag_key(item.tag) not in NON_CHARACTER_TAGS
        ]
        best_character_score = max((item.score for item in char_candidates), default=0.0)
        if truth_char and not pred_char:
            missed_scores.append(best_character_score)

        missing_char = sorted(truth_char - pred_char)
        extra_char = sorted(pred_char - truth_char)
        missing_gen = sorted(truth_gen - pred_gen)
        extra_gen = sorted(pred_gen - truth_gen)
        if missing_char or extra_char or missing_gen or extra_gen:
            failure_count += 1
            if len(failures) < max_failures:
                failures.append({
                    "file_name": row.get("file_name") or row.get("source_path"),
                    "character_missing": missing_char[:8],
                    "character_extra": extra_char[:8],
                    "general_missing": missing_gen[:20],
                    "general_extra": extra_gen[:20],
                    "best_character_score": round(best_character_score, 4),
                    "character_missing_flag": result.character_missing,
                    "tags_low_confidence": result.tags_low_confidence,
                })

    counts = tags_per_image
    return {
        "images": len(truth_rows),
        "character": {
            "tp": char_tp, "fp": char_fp, "fn": char_fn,
            **_prf(char_tp, char_fp, char_fn),
            "images_with_truth": char_images,
            "images_hit": char_hits,
            "image_hit_rate": round(char_hits / char_images, 4) if char_images else 0.0,
        },
        "general": {
            "tp": gen_tp, "fp": gen_fp, "fn": gen_fn,
            **_prf(gen_tp, gen_fp, gen_fn),
            "images_with_truth": gen_images,
            "images_hit": gen_hits,
            "image_hit_rate": round(gen_hits / gen_images, 4) if gen_images else 0.0,
        },
        "rating": {
            "counts": dict(ratings),
            "truth_count": rating_truth_total,
            "accuracy": round(rating_correct / rating_truth_total, 4) if rating_truth_total else 0.0,
            "confusion": dict(rating_confusion),
        },
        "low_confidence": {
            "tags_low_confidence": low_confidence,
            "character_missing": character_missing,
            "character_missing_with_truth": character_missing_with_truth,
            "missed_character_best_score": _score_summary(missed_scores),
            "missed_character_score_buckets": {
                "<0.30": sum(1 for score in missed_scores if score < 0.30),
                "0.30-0.45": sum(1 for score in missed_scores if 0.30 <= score < 0.45),
                "0.45-0.55": sum(1 for score in missed_scores if 0.45 <= score < 0.55),
                ">=0.55": sum(1 for score in missed_scores if score >= 0.55),
            },
        },
        "tags_per_image": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": round(sum(counts) / len(counts), 2) if counts else 0.0,
        },
        "failure_count": failure_count,
        "failures_truncated": failure_count > len(failures),
        "failures": failures,
    }


def calibrate_from_rows(
    predictions_per_image: Sequence[Sequence[TagPrediction]],
    truth_rows: Sequence[Mapping[str, Any]],
    config: TaggerConfig | None = None,
    *,
    character_labels: Iterable[str] | None = None,
    character_min_precision: float = 0.90,
    character_min_recall: float = 0.80,
) -> tuple[TaggerConfig, dict[str, Any]]:
    """Calibrate character/general thresholds from rows with known ``character``/``tags``.

    The returned report contains both sweeps, the selected config, and a full
    post-processing evaluation (per-category metrics, low-confidence counts and
    tag-level failures). ``character_labels`` are the checkpoint's character
    vocabulary, used to keep character identities out of general-tag truth.
    """
    config = config or TaggerConfig()
    if len(predictions_per_image) != len(truth_rows):
        raise ValueError("predictions_per_image and truth_rows must have the same length")
    character_label_keys = _truth_keys(character_labels or [])
    character_truth = [_character_truth(row, character_label_keys) for row in truth_rows]
    tag_truth = [_as_values(row.get("tags")) for row in truth_rows]
    character_report = calibrate_character(
        predictions_per_image, character_truth, max_characters=config.max_characters,
        min_precision=character_min_precision, min_recall=character_min_recall,
        preferred_threshold=0.55)
    general_excludes = []
    for row, truth in zip(truth_rows, character_truth):
        exclude = set(truth)
        exclude |= _truth_keys(_as_values(row.get("artist")) + _as_values(row.get("artists")))
        general_excludes.append(exclude)
    general_report = calibrate_general(
        predictions_per_image, tag_truth, max_tags=config.max_tags,
        exclude_per_image=general_excludes, exclude_labels=character_labels,
        preferred_threshold=0.35)
    calibrated = TaggerConfig(
        character_threshold=character_report["threshold"],
        general_threshold=general_report["threshold"],
        rating_threshold=config.rating_threshold,
        max_characters=config.max_characters,
        max_tags=config.max_tags,
        min_general_tags=config.min_general_tags,
    )
    evaluation = evaluate_tag_results(
        predictions_per_image, truth_rows, calibrated, character_labels=character_labels)
    return calibrated, {
        "character": character_report,
        "general": general_report,
        "evaluation": evaluation,
    }


# --------------------------------------------------------------------------- model wrappers


class TransformersTagger:
    """Lazy WD14/PixAI-style multi-label image tagger loaded from a local path.

    Models differ in label metadata. Pass ``category_map`` (normalized tag ->
    category) when the checkpoint does not encode categories; unrecognized
    labels default to ``general``. No Hub download is attempted by default.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        backend: str = "wd14",
        device: str = "cuda",
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        category_map: Mapping[str, str] | None = None,
        top_k: int | None = 200,
    ):
        if backend not in ("wd14", "pixai", "auto", "transformers"):
            raise ValueError(f"Unknown tagger backend {backend!r}")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1 or None")
        self.model_path = Path(model_path)
        self.backend = backend
        self.device = device
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.category_map = {normalize_tag_key(key): normalize_category(value)
                             for key, value in (category_map or {}).items()}
        self.top_k = top_k
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._labels: list[str] = []
        self._runtime = ""

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import transformers
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except Exception as error:  # pragma: no cover - optional dependency
            raise TaggerError(f"transformers/torch unavailable: {error}") from error
        kwargs = {"local_files_only": self.local_files_only,
                  "trust_remote_code": self.trust_remote_code}
        try:
            self._processor = AutoImageProcessor.from_pretrained(str(self.model_path), **kwargs)
            self._model = AutoModelForImageClassification.from_pretrained(
                str(self.model_path), **kwargs).eval().to(self.device)
        except Exception as error:
            raise TaggerError(
                f"Failed to load tagger {self.model_path} (local_files_only={self.local_files_only}; "
                f"models are never downloaded): {error}") from error
        self._torch = torch
        self._runtime = f"transformers {transformers.__version__}; torch {torch.__version__}"
        id2label = getattr(self._model.config, "id2label", {}) or {}
        self._labels = [str(id2label[index]) for index in range(len(id2label))]

    def _category(self, label: str) -> str:
        key = normalize_tag_key(label)
        if key in self.category_map:
            return self.category_map[key]
        return "general"

    def character_labels(self) -> list[str]:
        """Label names this checkpoint exposes as character identities."""
        self._ensure_loaded()
        return [label for label in self._labels if self._category(label) == "character"]

    def describe(self) -> dict[str, Any]:
        """Runtime metadata for calibration reports (never downloads)."""
        self._ensure_loaded()
        categories = Counter(self._category(label) for label in self._labels)
        return {
            "backend": "transformers",
            "model": str(self.model_path),
            "device": self.device,
            "runtime": self._runtime,
            "labels": len(self._labels),
            "categories": dict(categories),
            "local_files_only": self.local_files_only,
        }

    def predict_batch(self, images: Sequence[Image.Image]) -> list[list[TagPrediction]]:
        if not images:
            return []
        self._ensure_loaded()
        torch = self._torch
        assert torch is not None and self._processor is not None and self._model is not None
        prepared = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
        inputs = self._processor(images=prepared, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits.float()
        if not self._labels or logits.shape[-1] != len(self._labels):
            raise TaggerError(
                f"Tagger {self.model_path} exposes {logits.shape[-1]} logits but "
                f"{len(self._labels)} id2label entries; supply a label map for this checkpoint.")
        probabilities = torch.sigmoid(logits)
        results: list[list[TagPrediction]] = []
        for row in probabilities:
            scores, indices = torch.topk(row, k=min(self.top_k or len(row), len(row)))
            predictions = [
                TagPrediction(tag=self._labels[int(index)], score=float(score),
                              category=self._category(self._labels[int(index)]))
                for score, index in zip(scores.tolist(), indices.tolist())
                if int(index) < len(self._labels)
            ]
            results.append(predictions)
        return results

    def predict(self, image: Image.Image) -> list[TagPrediction]:
        return self.predict_batch([image])[0]


def _file_fingerprint(path: Path) -> dict[str, Any]:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _onnx_model_path(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser()
    if path.is_file() and path.suffix.lower() == ".onnx":
        return path
    if path.is_dir():
        preferred = path / "model.onnx"
        if preferred.is_file():
            return preferred
        candidates = sorted(item for item in path.glob("*.onnx") if item.is_file())
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise TaggerError(
                f"Multiple ONNX files in {path}; pass the exact .onnx file instead")
    raise TaggerError(
        f"ONNX tagger model not found: {path} (models are never downloaded; "
        "pass a local model.onnx or checkpoint directory)")


def _find_tags_csv(onnx_path: Path, tags_csv: str | Path | None) -> Path:
    if tags_csv is not None:
        path = Path(tags_csv).expanduser()
        if not path.is_file():
            raise TaggerError(f"Tag label CSV not found: {path} (models are never downloaded)")
        return path
    candidates = [onnx_path.with_suffix(".csv"), onnx_path.parent / "selected_tags.csv",
                  onnx_path.parent / "tags.csv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    csvs = sorted(item for item in onnx_path.parent.glob("*.csv") if item.is_file())
    selected = [item for item in csvs if item.name.lower() == "selected_tags.csv"]
    if selected:
        return selected[0]
    if len(csvs) == 1:
        return csvs[0]
    raise TaggerError(
        f"No tag label CSV next to {onnx_path}; pass --tags-csv (WD14 uses selected_tags.csv)")


def _find_preprocess(onnx_path: Path, preprocess: str | Path | None) -> Path | None:
    if preprocess is not None:
        path = Path(preprocess).expanduser()
        if not path.is_file():
            raise TaggerError(f"Preprocess config not found: {path}")
        return path
    candidate = onnx_path.parent / "preprocess.json"
    return candidate if candidate.is_file() else None


def _int_dim(value: Any) -> int | None:
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) and value > 0 else None


def _infer_onnx_layout(shape: Sequence[Any]) -> tuple[str, int]:
    dims = list(shape)
    if len(dims) != 4:
        raise TaggerError(f"Unsupported ONNX tagger input rank {len(dims)}: {dims}")
    if _int_dim(dims[-1]) == 3:
        size = _int_dim(dims[1]) or _int_dim(dims[2])
        if size is None:
            raise TaggerError(f"Cannot infer NHWC input size from shape {dims}")
        return "NHWC", size
    if _int_dim(dims[1]) == 3:
        size = _int_dim(dims[2]) or _int_dim(dims[3])
        if size is None:
            raise TaggerError(f"Cannot infer NCHW input size from shape {dims}")
        return "NCHW", size
    raise TaggerError(f"Cannot infer tagger input layout from shape {dims}")


def _resolve_onnx_providers(
    ort: Any,
    device: str,
    providers: Sequence[str] | None,
) -> tuple[list[str], list[dict[str, Any]], bool]:
    available = list(ort.get_available_providers())
    requested_gpu = not str(device).lower().startswith("cpu")
    if providers:
        chosen = [name for name in providers if name in available]
        if not chosen:
            raise TaggerError(
                f"None of the requested ONNX providers {list(providers)} are available: {available}")
    else:
        chosen = []
        if requested_gpu:
            for candidate in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
                if candidate in available:
                    chosen.append(candidate)
                    break
        if "CPUExecutionProvider" in available:
            chosen.append("CPUExecutionProvider")
        elif not chosen:
            chosen = available[:1]
    device_id = 0
    text = str(device)
    if ":" in text:
        try:
            device_id = int(text.rsplit(":", 1)[1])
        except ValueError:
            device_id = 0
    options: list[dict[str, Any]] = []
    for name in chosen:
        if name in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            options.append({"device_id": device_id})
        else:
            options.append({})
    return chosen, options, requested_gpu


def _apply_preprocess_stages(image: Image.Image, stages: Sequence[Mapping[str, Any]]) -> Any:
    """Minimal imgutils ``preprocess.json`` executor (PixAI/deepghs checkpoints).

    Supports convert_rgb, resize, center_crop, to_tensor/maybe_to_tensor and
    normalize, which covers the published deepghs tagger configs. Unknown
    stages fail loudly instead of silently producing wrong logits.
    """
    import numpy as np

    interpolations = {
        "nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS, "box": Image.BOX, "hamming": Image.HAMMING,
    }
    value: Any = ImageOps.exif_transpose(image).convert("RGB")
    for stage in stages:
        kind = str(stage.get("type", "")).lower()
        if kind == "convert_rgb":
            if isinstance(value, Image.Image):
                value = value.convert("RGB")
        elif kind == "resize":
            if not isinstance(value, Image.Image):
                raise TaggerError("preprocess resize must precede to_tensor")
            size = stage.get("size")
            width, height = value.size
            if isinstance(size, int) or (isinstance(size, (list, tuple)) and len(size) == 1):
                target = int(size if isinstance(size, int) else size[0])
                if width < height:
                    width, height = target, int(target * height / width)
                else:
                    width, height = int(target * width / height), target
            elif isinstance(size, (list, tuple)) and len(size) == 2:
                width, height = int(size[0]), int(size[1])
            else:
                raise TaggerError(f"Invalid preprocess resize size: {size!r}")
            value = value.resize((width, height), interpolations.get(
                str(stage.get("interpolation", "bilinear")).lower(), Image.BILINEAR))
        elif kind == "center_crop":
            if not isinstance(value, Image.Image):
                raise TaggerError("preprocess center_crop must precede to_tensor")
            size = stage.get("size")
            crop_w, crop_h = ((size, size) if isinstance(size, int)
                              else (int(size[0]), int(size[1])))
            width, height = value.size
            if width < crop_w or height < crop_h:
                canvas = Image.new("RGB", (max(width, crop_w), max(height, crop_h)), (0, 0, 0))
                canvas.paste(value, ((max(width, crop_w) - width) // 2,
                                     (max(height, crop_h) - height) // 2))
                value = canvas
            left = (value.width - crop_w) // 2
            top = (value.height - crop_h) // 2
            value = value.crop((left, top, left + crop_w, top + crop_h))
        elif kind in ("to_tensor", "maybe_to_tensor"):
            if isinstance(value, Image.Image):
                value = np.asarray(value, dtype=np.float32) / 255.0
                value = np.transpose(value, (2, 0, 1))
        elif kind == "normalize":
            array = np.asarray(value, dtype=np.float32)
            mean = np.asarray(stage.get("mean", 0.0), dtype=np.float32)
            std = np.asarray(stage.get("std", 1.0), dtype=np.float32)
            if array.ndim == 3 and mean.size > 1:
                mean = mean.reshape(-1, 1, 1)
            if array.ndim == 3 and std.size > 1:
                std = std.reshape(-1, 1, 1)
            value = (array - mean) / std
        else:
            raise TaggerError(f"Unsupported preprocess stage {kind!r} in preprocess.json")
    if isinstance(value, Image.Image):
        value = np.asarray(value, dtype=np.float32) / 255.0
        value = np.transpose(value, (2, 0, 1))
    return np.asarray(value, dtype=np.float32)


class OnnxTagger:
    """Lazy ONNX Runtime tagger for local WD14/PixAI checkpoints.

    WD14 v3 ONNX files use NHWC 448x448 float input with white padding and BGR
    channel order (the ComfyUI/SmilingWolf convention). Checkpoints that ship
    ``preprocess.json`` (deepghs PixAI conversions) use that pipeline instead.
    Loading is local-only; a missing file is an error, never a download.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        tags_csv: str | Path | None = None,
        preprocess: str | Path | None = None,
        device: str = "auto",
        providers: Sequence[str] | None = None,
        local_files_only: bool = True,
        category_map: Mapping[str, str] | None = None,
        top_k: int | None = 200,
        input_size: int | None = None,
    ):
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1 or None")
        if input_size is not None and input_size < 1:
            raise ValueError("input_size must be >= 1")
        self.model_path = Path(model_path)
        self.tags_csv = Path(tags_csv) if tags_csv is not None else None
        self.preprocess = Path(preprocess) if preprocess is not None else None
        self.device = device
        self.providers = tuple(providers) if providers else None
        self.local_files_only = local_files_only
        self.category_map = {normalize_tag_key(key): normalize_category(value)
                             for key, value in (category_map or {}).items()}
        self.top_k = top_k
        self.input_size = input_size
        self._session: Any | None = None
        self._np: Any | None = None
        self._labels: list[tuple[str, str]] = []
        self._input_name = "input"
        self._output_name = "output"
        self._layout = "NHWC"
        self._size = 448
        self._fixed_batch = False
        self._preprocess_stages: list[dict[str, Any]] | None = None
        self._description: dict[str, Any] = {}

    @property
    def backend(self) -> str:
        return "onnx"

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        try:
            import numpy as np
            import onnxruntime as ort
        except Exception as error:  # pragma: no cover - optional dependency
            raise TaggerError(f"onnxruntime/numpy unavailable: {error}") from error
        onnx_path = _onnx_model_path(self.model_path)
        csv_path = _find_tags_csv(onnx_path, self.tags_csv)
        preprocess_path = _find_preprocess(onnx_path, self.preprocess)
        labels = read_tag_labels(csv_path, category_map=self.category_map)
        chosen, options, requested_gpu = _resolve_onnx_providers(
            ort, self.device, self.providers)
        try:
            session = ort.InferenceSession(
                str(onnx_path), providers=chosen, provider_options=options)
        except Exception as error:
            raise TaggerError(
                f"Failed to load ONNX tagger {onnx_path} (local_files_only={self.local_files_only}; "
                f"models are never downloaded): {error}") from error
        inputs = session.get_inputs()
        if not inputs:
            raise TaggerError(f"ONNX tagger has no inputs: {onnx_path}")
        input_meta = inputs[0]
        layout, size = _infer_onnx_layout(input_meta.shape)
        if self.input_size is not None:
            size = self.input_size
        outputs = session.get_outputs()
        hinted = [item for item in outputs if item.name.lower() in
                  ("output", "probabilities", "probability", "prediction", "logits")]
        matching = [item for item in outputs
                    if list(item.shape) and _int_dim(item.shape[-1]) == len(labels)]
        hinted_matching = [item for item in hinted if item in matching]
        output_meta = (hinted_matching[0] if len(hinted_matching) == 1
                       else matching[0] if len(matching) == 1
                       else hinted[0] if len(hinted) == 1
                       else outputs[0] if len(outputs) == 1 else None)
        if output_meta is None:
            raise TaggerError(
                f"Cannot identify the tag output for {onnx_path}: outputs="
                f"{[(item.name, item.shape) for item in outputs]}, labels={len(labels)}")
        preprocess_stages = None
        if preprocess_path is not None:
            import json

            payload = json.loads(preprocess_path.read_text(encoding="utf-8"))
            stages = payload.get("stages", []) if isinstance(payload, dict) else payload
            if not isinstance(stages, list):
                raise TaggerError(f"preprocess.json has no stage list: {preprocess_path}")
            preprocess_stages = [dict(stage) for stage in stages]
        elif layout == "NCHW":
            raise TaggerError(
                f"{onnx_path} has an NCHW input but no preprocess.json; "
                "pass --preprocess or use the NHWC WD14 ONNX export")

        actual = list(session.get_providers())
        if requested_gpu and not any("CUDA" in name or "Tensorrt" in name for name in actual):
            import warnings

            warnings.warn(
                f"Requested GPU tagger inference but ONNX Runtime selected {actual}; "
                "falling back to CPU. Check the onnxruntime-gpu and CUDA library paths.",
                RuntimeWarning, stacklevel=2)
        self._session = session
        self._np = np
        self._labels = labels
        self._input_name = input_meta.name
        self._output_name = output_meta.name
        self._layout = layout
        self._size = size
        self._preprocess_stages = preprocess_stages
        self._fixed_batch = (
            _int_dim(list(input_meta.shape)[0] if input_meta.shape else None) == 1
            or _int_dim(list(output_meta.shape)[0] if output_meta.shape else None) == 1
        )
        self._description = {
            "backend": "onnx",
            "model": str(onnx_path),
            "model_bytes": onnx_path.stat().st_size,
            "model_sha256": _file_fingerprint(onnx_path)["sha256"],
            "tags_csv": str(csv_path),
            "tags_bytes": csv_path.stat().st_size,
            "tags_sha256": _file_fingerprint(csv_path)["sha256"],
            "preprocess": str(preprocess_path) if preprocess_path else None,
            "runtime": f"onnxruntime {ort.__version__}",
            "providers": actual,
            "device": self.device,
            "input": {"name": input_meta.name, "shape": list(input_meta.shape),
                      "layout": layout, "size": size},
            "labels": len(labels),
            "categories": dict(Counter(category for _, category in labels)),
            "local_files_only": self.local_files_only,
            "output_activation": "unknown",
        }

    def character_labels(self) -> list[str]:
        """Label names this checkpoint exposes as character identities."""
        self._ensure_loaded()
        return [name for name, category in self._labels if category == "character"]

    def describe(self) -> dict[str, Any]:
        """Runtime metadata for calibration reports (never downloads)."""
        self._ensure_loaded()
        return dict(self._description)

    def _prepare_image(self, image: Image.Image) -> Any:
        np = self._np
        assert np is not None
        if self._preprocess_stages is not None:
            array = _apply_preprocess_stages(image, self._preprocess_stages)
            if array.ndim != 3:
                raise TaggerError(f"preprocess produced rank {array.ndim}, expected 3")
            if self._layout == "NHWC":
                array = np.transpose(array, (1, 2, 0))
            return np.ascontiguousarray(array[None, ...], dtype=np.float32)
        if self._layout == "NHWC":
            prepared = ImageOps.exif_transpose(image).convert("RGB")
            width, height = prepared.size
            ratio = self._size / max(width, height)
            resized = prepared.resize(
                (max(1, int(width * ratio)), max(1, int(height * ratio))), Image.LANCZOS)
            canvas = Image.new("RGB", (self._size, self._size), (255, 255, 255))
            canvas.paste(resized, ((self._size - resized.width) // 2,
                                   (self._size - resized.height) // 2))
            array = np.asarray(canvas, dtype=np.float32)[:, :, ::-1]
            return np.ascontiguousarray(array[None, ...])
        raise TaggerError(f"Unsupported layout {self._layout} without preprocess.json")

    def predict_batch(self, images: Sequence[Image.Image]) -> list[list[TagPrediction]]:
        if not images:
            return []
        self._ensure_loaded()
        if self._fixed_batch and len(images) > 1:
            results: list[list[TagPrediction]] = []
            for image in images:
                results.extend(self.predict_batch([image]))
            return results
        np = self._np
        assert np is not None and self._session is not None
        batch = np.concatenate([self._prepare_image(image) for image in images], axis=0)
        raw = np.asarray(
            self._session.run([self._output_name], {self._input_name: batch})[0],
            dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.ndim != 2 or raw.shape[0] != len(images) or raw.shape[1] != len(self._labels):
            raise TaggerError(
                f"ONNX tagger returned {raw.shape}, expected ({len(images)}, {len(self._labels)})")
        if not np.isfinite(raw).all():
            raise TaggerError("ONNX tagger returned non-finite scores")
        if float(raw.min()) < 0.0 or float(raw.max()) > 1.0:
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw, -60.0, 60.0)))
            self._description["output_activation"] = "sigmoid"
        else:
            probabilities = raw
            self._description["output_activation"] = "identity"
        probabilities = np.clip(probabilities, 0.0, 1.0)
        k = len(self._labels) if self.top_k is None else min(self.top_k, len(self._labels))
        results: list[list[TagPrediction]] = []
        for row in probabilities:
            indices = np.argpartition(-row, k - 1)[:k]
            ranked = sorted(
                ((float(row[index]), self._labels[int(index)][0],
                  self._labels[int(index)][1]) for index in indices),
                key=lambda item: (-item[0], normalize_tag_key(item[1])))
            results.append([TagPrediction(tag=name, score=score, category=category)
                            for score, name, category in ranked])
        return results

    def predict(self, image: Image.Image) -> list[TagPrediction]:
        return self.predict_batch([image])[0]


def _is_onnx_model(model_path: str | Path) -> bool:
    path = Path(model_path)
    return (path.suffix.lower() == ".onnx"
            or (path.is_dir() and any(path.glob("*.onnx"))))


def load_tagger(
    model_path: str | Path,
    *,
    backend: str = "auto",
    device: str = "auto",
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    category_map: Mapping[str, str] | None = None,
    tags_csv: str | Path | None = None,
    preprocess: str | Path | None = None,
    providers: Sequence[str] | None = None,
    top_k: int | None = 200,
) -> TransformersTagger | OnnxTagger:
    """Construct a lazy tagger; model loading happens on the first prediction.

    ``auto`` picks ONNX Runtime for a local ``.onnx`` file/directory and
    Transformers for a saved Hugging Face checkpoint directory. ``wd14`` and
    ``pixai`` also prefer ONNX when such a file is present, preserving the
    old Transformers behavior for local checkpoint directories.
    """
    backend = str(backend).lower()
    if backend not in ("wd14", "pixai", "auto", "onnx", "transformers"):
        raise ValueError(f"Unknown tagger backend {backend!r}")
    use_onnx = backend == "onnx" or (
        backend != "transformers" and _is_onnx_model(model_path))
    if use_onnx:
        return OnnxTagger(
            model_path, tags_csv=tags_csv, preprocess=preprocess, device=device,
            providers=providers, local_files_only=local_files_only,
            category_map=category_map, top_k=top_k)
    return TransformersTagger(
        model_path, backend=backend, device=device, local_files_only=local_files_only,
        trust_remote_code=trust_remote_code, category_map=category_map, top_k=top_k)


# --------------------------------------------------------------------------- row driver


def _image_path(row: Mapping[str, Any], image_root: Path) -> Path:
    source = row.get("source_path")
    if isinstance(source, str) and source:
        path = Path(source)
        return path if path.is_absolute() else image_root / path
    file_name = row.get("file_name")
    if isinstance(file_name, str) and file_name:
        return image_root / file_name
    raise TaggerError("metadata row has neither source_path nor file_name")


def predict_metadata_rows(
    rows: Sequence[Mapping[str, Any]],
    predictor: Any,
    *,
    image_root: str | Path,
    batch_size: int = 8,
    progress: bool = True,
) -> tuple[list[list[TagPrediction]], list[dict[str, Any]]]:
    """Raw predictions aligned with loadable rows (used by calibration)."""
    predict_batch = getattr(predictor, "predict_batch", None)
    predict = getattr(predictor, "predict", None)
    if not callable(predict_batch) and not callable(predict):
        raise TaggerError("predictor must expose predict_batch(images) or predict(image)")
    root = Path(image_root)
    predictions_all: list[list[TagPrediction]] = []
    valid_rows: list[dict[str, Any]] = []
    starts = list(range(0, len(rows), max(1, batch_size)))
    if progress:
        from tqdm import tqdm

        starts = list(tqdm(starts, desc="tagger", unit="batch"))
    for start in starts:
        chunk = rows[start:start + max(1, batch_size)]
        images: list[Image.Image] = []
        chunk_rows: list[dict[str, Any]] = []
        for row in chunk:
            try:
                path = _image_path(row, root)
                with Image.open(path) as opened:
                    images.append(ImageOps.exif_transpose(opened).convert("RGB"))
                chunk_rows.append(dict(row))
            except Exception:
                continue
        if not images:
            continue
        try:
            predictions = (predict_batch(images) if callable(predict_batch)
                           else [predict(image) for image in images])
        finally:
            for image in images:
                image.close()
        predictions_all.extend(predictions)
        valid_rows.extend(chunk_rows)
    return predictions_all, valid_rows


def tag_metadata_rows(
    rows: Sequence[Mapping[str, Any]],
    predictor: Any,
    *,
    image_root: str | Path,
    config: TaggerConfig | None = None,
    character_vocab: Mapping[str, str] | Iterable[str] | None = None,
    general_vocab: Mapping[str, str] | Iterable[str] | None = None,
    batch_size: int = 8,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run predictions in batches and write tagger fields into copies of ``rows``."""
    config = config or TaggerConfig()
    if character_vocab is not None and not isinstance(character_vocab, Mapping):
        character_vocab = build_vocabulary(character_vocab)
    if general_vocab is not None and not isinstance(general_vocab, Mapping):
        general_vocab = build_vocabulary(general_vocab)
    predict_batch = getattr(predictor, "predict_batch", None)
    predict = getattr(predictor, "predict", None)
    if not callable(predict_batch) and not callable(predict):
        raise TaggerError("predictor must expose predict_batch(images) or predict(image)")

    output = [dict(row) for row in rows]
    root = Path(image_root)
    report: dict[str, Any] = {
        "rows": len(rows), "tagged": 0, "errors": [], "character_found": 0,
        "character_missing": 0, "tags_low_confidence": 0,
        "ratings": Counter(), "new_characters": Counter(), "tag_counts": [],
    }
    iterator: Iterable[int] = range(0, len(output), max(1, batch_size))
    if progress:
        from tqdm import tqdm

        iterator = tqdm(list(iterator), desc="tagger", unit="batch")
    for start in iterator:
        chunk = output[start:start + max(1, batch_size)]
        images: list[Image.Image] = []
        indices: list[int] = []
        for offset, row in enumerate(chunk):
            try:
                path = _image_path(row, root)
                with Image.open(path) as opened:
                    images.append(ImageOps.exif_transpose(opened).convert("RGB"))
                indices.append(start + offset)
            except Exception as error:
                report["errors"].append({"file_name": row.get("file_name"),
                                         "error": f"{type(error).__name__}: {error}"})
        if not images:
            continue
        try:
            predictions = (predict_batch(images) if callable(predict_batch)
                           else [predict(image) for image in images])
        finally:
            for image in images:
                image.close()
        for index, row_predictions in zip(indices, predictions):
            result = postprocess_tags(row_predictions, config,
                                       character_vocab=character_vocab,
                                       general_vocab=general_vocab)
            output[index].update(result.row_fields())
            report["tagged"] += 1
            report["character_found"] += int(bool(result.character))
            report["character_missing"] += int(not result.character)
            report["tags_low_confidence"] += int(result.tags_low_confidence)
            if result.rating:
                report["ratings"][result.rating] += 1
            report["new_characters"].update(result.new_character)
            report["tag_counts"].append(len(result.tags))
    report["ratings"] = dict(report["ratings"])
    report["new_characters"] = dict(report["new_characters"])
    counts = report.pop("tag_counts")
    report["tags_per_image"] = {
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "mean": round(sum(counts) / len(counts), 2) if counts else 0.0,
    }
    return output, report


# --------------------------------------------------------------------------- CLI


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _load_vocab(path: str | None) -> dict[str, str] | list[str]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [str(value) for value in payload]
    raise ValueError(f"vocabulary must be a JSON list or object: {path}")


def _runtime_config(
    config: TaggerConfig,
    predictor: Any,
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    description = predictor.describe() if hasattr(predictor, "describe") else {}
    calibration_summary = None
    if calibration:
        calibration_summary = {
            "character_threshold": calibration.get("character", {}).get("threshold"),
            "general_threshold": calibration.get("general", {}).get("threshold"),
            "character": {key: calibration.get("character", {}).get(key)
                          for key in ("precision", "recall", "f1", "meets_gates")},
            "general": {key: calibration.get("general", {}).get(key)
                        for key in ("precision", "recall", "f1")},
        }
    return {
        **config.to_dict(),
        "backend": description.get("backend"),
        "model": description.get("model"),
        "tags_csv": description.get("tags_csv"),
        "device": description.get("device"),
        "providers": description.get("providers"),
        "model_sha256": description.get("model_sha256"),
        "tags_sha256": description.get("tags_sha256"),
        "runtime": description.get("runtime"),
        "label_count": description.get("labels"),
        "input_size": (description.get("input") or {}).get("size"),
        "input_layout": (description.get("input") or {}).get("layout"),
        "calibration": calibration_summary,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys
    from datetime import datetime, timezone

    from moss_lora.utils import atomic_write_json

    parser = argparse.ArgumentParser(
        prog="moss-lora-tag",
        description="Post-process PixAI/WD14 tagger predictions for staged metadata.")
    parser.add_argument("--input", default=None, help="metadata_staging.jsonl")
    parser.add_argument("--output", default=None, help="Tagged JSONL (default: metadata_tagged.jsonl)")
    parser.add_argument("--images", default=None,
                        help="Root for relative file_name values (default: input's parent)")
    parser.add_argument("--model", required=True,
                        help="Local tagger file or checkpoint directory (never downloaded)")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "onnx", "wd14", "pixai", "transformers"])
    parser.add_argument("--tags-csv", default=None,
                        help="Local selected_tags.csv (default: found next to model.onnx)")
    parser.add_argument("--preprocess", default=None,
                        help="Local preprocess.json for NCHW ONNX checkpoints")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--providers", default=None,
                        help="ONNX providers, e.g. CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=200,
                        help="Scores kept per image; <=0 keeps every label (use for calibration)")
    parser.add_argument("--config", default=None, help="Frozen tagger_config.json")
    parser.add_argument("--character-vocab", default=None, help="JSON list/map of known characters")
    parser.add_argument("--general-vocab", default=None, help="JSON list/map of known general tags")
    parser.add_argument("--category-map", default=None, help="JSON tag -> category map")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--calibrate-jsonl", default=None,
                        help="Current metadata with known character/tags (200-image calibration)")
    parser.add_argument("--calibration-images", default=None,
                        help="Image root for --calibrate-jsonl (default: calibration JSONL's parent)")
    parser.add_argument("--calibration-limit", type=int, default=200,
                        help="Calibration row limit; <=0 uses every row")
    parser.add_argument("--calibration-report", default=None,
                        help="Calibration report JSON (default: output dir/tagger_calibration.json)")
    parser.add_argument("--character-min-precision", type=float, default=0.90)
    parser.add_argument("--character-min-recall", type=float, default=0.80)
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Calibrate and write config/report without tagging --input")
    parser.add_argument("--limit", type=int, default=None, help="Pilot row limit")
    args = parser.parse_args(argv)

    if args.calibrate_only and not args.calibrate_jsonl:
        parser.error("--calibrate-only requires --calibrate-jsonl")
    if not args.calibrate_only and not args.input:
        parser.error("--input is required unless --calibrate-only is used")

    input_path = Path(args.input) if args.input else None
    rows = _read_jsonl(input_path) if input_path else []
    if args.limit is not None:
        rows = rows[: args.limit]
    image_root = (Path(args.images) if args.images
                  else input_path.parent if input_path else Path.cwd())
    output_path = (Path(args.output) if args.output
                   else input_path.with_name("metadata_tagged.jsonl") if input_path else None)

    config = TaggerConfig.load(args.config) if args.config else TaggerConfig()
    category_map = (json.loads(Path(args.category_map).read_text(encoding="utf-8"))
                    if args.category_map else None)
    character_vocab = _load_vocab(args.character_vocab)
    general_vocab = _load_vocab(args.general_vocab)
    providers = ([item.strip() for item in args.providers.split(",") if item.strip()]
                 if args.providers else None)
    predictor = load_tagger(
        args.model, backend=args.backend, device=args.device,
        tags_csv=args.tags_csv, preprocess=args.preprocess, providers=providers,
        trust_remote_code=args.trust_remote_code, category_map=category_map,
        top_k=None if args.top_k <= 0 else args.top_k)

    calibration_report = None
    calibration_payload = None
    if args.calibrate_jsonl:
        calibration_path = Path(args.calibrate_jsonl)
        calibration_rows_all = _read_jsonl(calibration_path)
        if not character_vocab:
            character_vocab = sorted({
                str(value)
                for row in calibration_rows_all
                for value in _as_values(row.get("character")) + _as_values(row.get("characters"))
                if isinstance(value, str) and value.strip()
            })
        calibration_rows = (calibration_rows_all if args.calibration_limit <= 0
                            else calibration_rows_all[: args.calibration_limit])
        calibration_root = (Path(args.calibration_images) if args.calibration_images
                            else calibration_path.parent)
        predictions, valid_rows = predict_metadata_rows(
            calibration_rows, predictor, image_root=calibration_root, batch_size=args.batch_size)
        if not valid_rows:
            raise TaggerError(
                f"No calibration images could be read from {calibration_path} "
                f"(image root: {calibration_root})")
        character_labels = (predictor.character_labels()
                            if hasattr(predictor, "character_labels") else None)
        config, calibration_report = calibrate_from_rows(
            predictions, valid_rows, config, character_labels=character_labels,
            character_min_precision=args.character_min_precision,
            character_min_recall=args.character_min_recall)
        calibration_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": config.to_dict(),
            "model": predictor.describe() if hasattr(predictor, "describe") else {},
            "calibration": calibration_report,
            "requested": {
                "backend": args.backend,
                "device": args.device,
                "providers": providers,
                "batch_size": args.batch_size,
                "top_k": args.top_k,
                "rows": len(valid_rows),
                "calibration_limit": args.calibration_limit,
                "character_min_precision": args.character_min_precision,
                "character_min_recall": args.character_min_recall,
            },
        }
        if args.calibration_report:
            report_path = Path(args.calibration_report)
        elif output_path is not None:
            report_path = output_path.parent / "tagger_calibration.json"
        else:
            report_path = calibration_path.parent / "tagger_calibration.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report_path, calibration_payload)
        config_path = report_path.parent / "tagger_config.json"
        atomic_write_json(config_path, _runtime_config(config, predictor, calibration_report))
        selected = calibration_report["evaluation"]
        print(json.dumps({
            "calibration_report": str(report_path),
            "tagger_config": str(config_path),
            "character_threshold": config.character_threshold,
            "general_threshold": config.general_threshold,
            "character": selected["character"],
            "general": selected["general"],
            "low_confidence": selected["low_confidence"],
        }, ensure_ascii=False, indent=2), file=sys.stderr)

    if args.calibrate_only:
        return

    assert input_path is not None and output_path is not None
    tagged_rows, report = tag_metadata_rows(
        rows, predictor, image_root=image_root, config=config,
        character_vocab=character_vocab or None, general_vocab=general_vocab or None,
        batch_size=args.batch_size)
    _write_jsonl(output_path, tagged_rows)

    output_dir = output_path.parent
    atomic_write_json(output_dir / "tagger_config.json",
                      _runtime_config(config, predictor, calibration_report))
    atomic_write_json(output_dir / "tagger_report.json", report)
    print(json.dumps({"output": str(output_path), "tagged": report["tagged"],
                      "character_found": report["character_found"],
                      "character_missing": report["character_missing"]},
                     ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
