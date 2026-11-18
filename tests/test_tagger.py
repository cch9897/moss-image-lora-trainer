"""Tagger post-processing, vocabulary mapping, calibration and batch driver."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from moss_lora.tagger import (
    NON_CHARACTER_TAGS,
    OnnxTagger,
    TaggerConfig,
    TaggerError,
    TagPrediction,
    TransformersTagger,
    build_vocabulary,
    calibrate_character,
    calibrate_from_rows,
    calibrate_general,
    evaluate_tag_results,
    load_tagger,
    postprocess_tags,
    read_tag_labels,
    tag_metadata_rows,
)
from tests.helpers import gradient_image


def predictions() -> list[TagPrediction]:
    return [
        TagPrediction("hero", 0.91, "character"),
        TagPrediction("villager", 0.50, "character"),
        TagPrediction("1girl", 0.99, "general"),
        TagPrediction("city_street", 0.40, "general"),
        TagPrediction("solo", 0.90, "general"),
        TagPrediction("watermark", 0.90, "general"),
        TagPrediction("centered", 0.30, "general"),
        TagPrediction("general", 0.80, "rating"),
    ]


def test_postprocess_thresholds_vocab_and_negatives():
    result = postprocess_tags(
        predictions(), TaggerConfig(character_threshold=0.55, general_threshold=0.35),
        character_vocab={"hero": "Hero"}, general_vocab={"city_street": "city street"})
    assert result.character == ("Hero",)
    assert not result.character_missing
    assert result.new_character == ()
    assert "Hero" in result.tags and "city street" in result.tags
    assert "villager" not in result.character  # below character threshold
    assert "centered" not in result.tags  # below general threshold
    assert "watermark" not in result.tags  # boilerplate negative
    assert result.rating == "general"
    assert result.tags_low_confidence  # only solo + city street are general category
    assert result.tag_confidence["Hero"] == pytest.approx(0.91)


def test_postprocess_character_cap_order_and_new_character():
    many = [TagPrediction(f"char{index}", 0.9 - index * 0.05, "character") for index in range(6)]
    result = postprocess_tags(many, TaggerConfig(character_threshold=0.5, max_characters=4))
    assert result.character == ("char0", "char1", "char2", "char3")
    assert result.new_character == ("char0", "char1", "char2", "char3")


def test_postprocess_never_treats_structural_tags_as_characters():
    result = postprocess_tags(
        [TagPrediction(tag, 0.99, "character") for tag in ("1girl", "solo", "multiple girls")])
    assert result.character == () and result.character_missing
    assert NON_CHARACTER_TAGS >= {"1girl", "solo", "multiple girls"}


def test_zero_characters_and_rating_threshold():
    result = postprocess_tags(
        [TagPrediction("general", 0.40, "rating"),
         TagPrediction("questionable", 0.20, "rating")],
        TaggerConfig(rating_threshold=0.5))
    assert result.character_missing and result.rating is None
    assert result.tags_low_confidence


def test_build_vocabulary_normalizes_keys():
    vocab = build_vocabulary(["Hakui_Koyori", "hakui koyori", "Another"])
    assert vocab["hakui koyori"] == "Hakui_Koyori"
    assert len(vocab) == 2


def test_calibration_character_and_general():
    character_predictions = [
        [TagPrediction("hero", 0.90, "character"), TagPrediction("decoy", 0.20, "character")],
        [TagPrediction("hero", 0.70, "character")],
    ]
    character_truth = [["hero"], ["hero"]]
    report = calibrate_character(character_predictions, character_truth,
                                 thresholds=[0.3, 0.5, 0.8], max_characters=4,
                                 preferred_threshold=None)
    assert report["threshold"] == 0.5
    assert report["precision"] == 1.0 and report["recall"] == 1.0
    assert report["meets_gates"]

    general_predictions = [
        [TagPrediction("tag a", 0.80, "general"), TagPrediction("tag b", 0.20, "general")],
        [TagPrediction("tag a", 0.70, "general")],
    ]
    general_truth = [["tag a"], ["tag a"]]
    report = calibrate_general(general_predictions, general_truth, thresholds=[0.3, 0.7],
                               preferred_threshold=None)
    assert report["threshold"] == 0.7 and report["f1"] == 1.0


def test_calibrate_from_rows_updates_config():
    predictions = [[TagPrediction("hero", 0.9, "character"), TagPrediction("tag", 0.8, "general")]]
    rows = [{"character": ["hero"], "tags": ["tag"]}]
    config, report = calibrate_from_rows(predictions, rows)
    assert config.character_threshold == report["character"]["threshold"]
    assert config.general_threshold == report["general"]["threshold"]
    assert config.rating_threshold == 0.50


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict_batch(self, images):
        self.calls += 1
        return [[TagPrediction("hero", 0.9, "character"),
                 TagPrediction("city street", 0.6, "general"),
                 TagPrediction("1girl", 0.95, "general"),
                 TagPrediction("solo", 0.9, "general"),
                 TagPrediction("outdoors", 0.5, "general"),
                 TagPrediction("sky", 0.4, "general")] for _ in images]


def test_tag_metadata_rows_updates_fields(tmp_path):
    image = tmp_path / "a.png"
    gradient_image(64, 64, 1).save(image)
    rows = [{"file_name": "a.png"}, {"file_name": "missing.png"}]
    tagged, report = tag_metadata_rows(rows, FakePredictor(), image_root=tmp_path,
                                       batch_size=2, progress=False)
    assert report["tagged"] == 1
    assert report["character_found"] == 1 and report["character_missing"] == 0
    assert report["ratings"] == {}
    assert len(report["errors"]) == 1
    assert tagged[0]["character"] == ["hero"]
    assert "city street" in tagged[0]["tags"] and "watermark" not in tagged[0]["tags"]
    assert tagged[0]["tags_low_confidence"] is False
    assert "tags" not in tagged[1] or tagged[1]["tags"] == []


def test_transformers_tagger_is_lazy_and_never_downloads(tmp_path):
    tagger = TransformersTagger(tmp_path / "missing-model", local_files_only=True)
    assert tagger._model is None
    with pytest.raises(TaggerError, match="never downloaded"):
        tagger.predict(gradient_image(32, 32, 1))


def test_tagger_config_roundtrip(tmp_path):
    config = TaggerConfig(character_threshold=0.6, general_threshold=0.4)
    path = tmp_path / "tagger_config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert TaggerConfig.load(path) == config
    with pytest.raises(ValueError):
        TaggerConfig.from_dict({"unknown": 1})
    with pytest.raises(ValueError):
        TaggerConfig(character_threshold=2.0)


def test_json_object_vocabulary_keeps_alias_keys(tmp_path):
    from moss_lora.tagger import _load_vocab

    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"hero_alias": "Hero"}), encoding="utf-8")
    vocab = _load_vocab(str(path))
    assert vocab == {"hero_alias": "Hero"}
    result = postprocess_tags([TagPrediction("hero_alias", 0.9, "character")],
                              character_vocab=vocab)
    assert result.character == ("Hero",)


def test_read_tag_labels_wd14_categories_and_override(tmp_path):
    path = tmp_path / "selected_tags.csv"
    path.write_text(
        "tag_id,name,category,count\n"
        "1,1girl,0,10\n"
        "2,hero,4,5\n"
        "3,general,9,4\n",
        encoding="utf-8",
    )
    assert read_tag_labels(path) == [
        ("1girl", "general"), ("hero", "character"), ("general", "rating")]
    assert read_tag_labels(path, category_map={"1girl": "meta"})[0] == ("1girl", "meta")
    with pytest.raises(TaggerError, match="not found"):
        read_tag_labels(tmp_path / "missing.csv")


def test_tagger_config_load_ignores_runtime_metadata(tmp_path):
    path = tmp_path / "tagger_config.json"
    path.write_text(json.dumps({
        "character_threshold": 0.6, "general_threshold": 0.4,
        "backend": "onnx", "model": "/models/tagger.onnx",
        "model_sha256": "abc", "calibration": {"character": 0.6},
    }), encoding="utf-8")
    assert TaggerConfig.load(path) == TaggerConfig(
        character_threshold=0.6, general_threshold=0.4)
    with pytest.raises(ValueError):
        TaggerConfig.from_dict({"unknown": 1})


def test_general_calibration_excludes_character_truth():
    predictions = [[TagPrediction("tag", 0.9, "general"),
                    TagPrediction("hero", 0.9, "general")]]
    report = calibrate_general(
        predictions, [["hero", "tag"]], thresholds=[0.5], exclude_per_image=[{"hero"}])
    assert report["precision"] == 0.5
    assert report["recall"] == 1.0


def test_evaluate_tag_results_reports_failures_and_low_confidence():
    predictions = [[
        TagPrediction("hero", 0.9, "character"),
        TagPrediction("wrong", 0.8, "character"),
        TagPrediction("1girl", 0.9, "general"),
    ]]
    rows = [{"file_name": "a.jpg", "character": ["hero"],
             "tags": ["hero", "1girl", "outdoors"]}]
    report = evaluate_tag_results(predictions, rows, character_labels=["hero"])
    assert report["character"]["precision"] == 0.5
    assert report["character"]["recall"] == 1.0
    assert report["general"]["precision"] == 1.0
    assert report["general"]["recall"] == 0.5
    assert report["low_confidence"]["tags_low_confidence"] == 1
    assert report["low_confidence"]["character_missing_with_truth"] == 0
    assert report["failure_count"] == 1
    failure = report["failures"][0]
    assert failure["file_name"] == "a.jpg"
    assert failure["character_extra"] == ["wrong"]
    assert failure["general_missing"] == ["outdoors"]
    assert "caption" not in failure


class _FakeInput:
    name = "input"

    def __init__(self, shape):
        self.shape = shape


class _FakeOutput:
    name = "output"
    shape = ["batch", 3]


class _FakeSession:
    input_shape = ["batch", 16, 16, 3]
    last_batch = None

    def __init__(self, path, providers=None, provider_options=None):
        self.path = path
        self.providers = providers or ["CPUExecutionProvider"]

    def get_inputs(self):
        return [_FakeInput(self.input_shape)]

    def get_outputs(self):
        return [_FakeOutput()]

    def get_providers(self):
        return self.providers

    def run(self, names, feed):
        _FakeSession.last_batch = feed["input"]
        return [np.asarray([[0.9, 0.4, 0.2], [0.1, 0.8, 0.3]], dtype=np.float32)]


def test_onnx_tagger_local_batch_preprocessing_and_categories(tmp_path, monkeypatch):
    onnxruntime = pytest.importorskip("onnxruntime")

    (tmp_path / "model.onnx").write_bytes(b"not-a-real-onnx")
    (tmp_path / "selected_tags.csv").write_text(
        "tag_id,name,category,count\n"
        "1,1girl,0,10\n"
        "2,hero,4,5\n"
        "3,general,9,4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(onnxruntime, "InferenceSession", _FakeSession)
    tagger = OnnxTagger(tmp_path / "model.onnx", device="cpu", top_k=3)
    assert tagger._session is None
    red = Image.new("RGB", (8, 4), (255, 0, 0))
    blue = Image.new("RGB", (4, 8), (0, 0, 255))
    predictions = tagger.predict_batch([red, blue])
    assert len(predictions) == 2
    assert [item.category for item in predictions[0]] == ["general", "character", "rating"]
    assert predictions[0][0].tag == "1girl" and predictions[0][0].score > 0.89
    batch = _FakeSession.last_batch
    assert batch.shape == (2, 16, 16, 3)
    assert np.allclose(batch[0, 0, 0], (255, 255, 255))  # white padding
    assert np.allclose(batch[0, 8, 8], (0, 0, 255))     # red -> BGR
    assert np.allclose(batch[1, 8, 8], (255, 0, 0))     # blue -> BGR
    description = tagger.describe()
    assert description["backend"] == "onnx"
    assert description["providers"] == ["CPUExecutionProvider"]
    assert description["categories"] == {"general": 1, "character": 1, "rating": 1}
    assert description["input"]["layout"] == "NHWC"
    assert tagger.character_labels() == ["hero"]


def test_onnx_tagger_requires_preprocess_for_nchw(tmp_path, monkeypatch):
    onnxruntime = pytest.importorskip("onnxruntime")

    (tmp_path / "model.onnx").write_bytes(b"not-a-real-onnx")
    (tmp_path / "selected_tags.csv").write_text(
        "tag_id,name,category,count\n1,hero,4,1\n", encoding="utf-8")

    class NchwSession(_FakeSession):
        input_shape = ["batch", 3, 16, 16]

    monkeypatch.setattr(onnxruntime, "InferenceSession", NchwSession)
    tagger = OnnxTagger(tmp_path / "model.onnx", device="cpu")
    with pytest.raises(TaggerError, match="NCHW"):
        tagger.predict(Image.new("RGB", (8, 8), (0, 0, 0)))


def test_onnx_tagger_is_lazy_and_never_downloads(tmp_path):
    tagger = OnnxTagger(tmp_path / "missing.onnx", device="cpu")
    assert tagger._session is None
    with pytest.raises(TaggerError, match="never downloaded"):
        tagger.predict(Image.new("RGB", (8, 8), (0, 0, 0)))


def test_load_tagger_auto_detects_onnx_and_transformers(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    assert isinstance(load_tagger(tmp_path / "model.onnx", backend="auto"), OnnxTagger)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    assert isinstance(load_tagger(checkpoint, backend="auto"), TransformersTagger)
    assert isinstance(load_tagger(tmp_path / "missing.onnx", backend="auto"), OnnxTagger)


def test_calibration_prefers_default_threshold_within_tie_tolerance():
    predictions = [[TagPrediction("tag a", 0.8, "general"),
                    TagPrediction("tag b", 0.2, "general")]]
    report = calibrate_general(
        predictions, [["tag a"]], thresholds=[0.3, 0.7], preferred_threshold=0.35)
    assert report["f1"] == 1.0
    assert report["threshold"] == 0.3


def test_general_calibration_ignores_rating_truth():
    predictions = [[TagPrediction("tag", 0.9, "general")]]
    report = calibrate_general(predictions, [["sensitive", "tag"]], thresholds=[0.5])
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0


def test_character_truth_recovers_tags_when_field_is_empty():
    predictions = [[TagPrediction("hero", 0.9, "character")]]
    rows = [{"file_name": "a.jpg", "character": [], "tags": ["hero", "1girl"]}]
    report = evaluate_tag_results(predictions, rows, character_labels=["hero"])
    assert report["character"]["precision"] == 1.0
    assert report["character"]["recall"] == 1.0
    assert report["character"]["images_with_truth"] == 1


def test_general_evaluation_excludes_capped_character_candidates():
    labels = [f"char{index}" for index in range(6)]
    predictions = [[TagPrediction(label, 0.9, "character") for label in labels]
                   + [TagPrediction("1girl", 0.9, "general")]]
    rows = [{"file_name": "a.jpg", "character": [], "tags": ["1girl"]}]
    report = evaluate_tag_results(
        predictions, rows, TaggerConfig(max_characters=2), character_labels=labels)
    assert report["general"]["fp"] == 0
    assert report["general"]["tp"] == 1
