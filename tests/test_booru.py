"""Booru lookup tests: raw-byte MD5, post parsing, tag mapping, HTTP retry policy, cache."""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
from pathlib import Path
from typing import Mapping

import pytest

from moss_lora.booru import (
    SITE_DANBOORU,
    STATUS_AUTH_BLOCKED,
    STATUS_ERROR,
    STATUS_HIT,
    STATUS_MISS,
    TERMINAL_STATUSES,
    BooruCache,
    DanbooruClient,
    HttpResponse,
    file_md5,
    load_vocab,
    map_post_tags,
    normalize_rating,
    parse_post,
    split_tag_string,
)

HIT_MD5 = "fa9f36a9f75aaf65206848489317a1df"  # md5(b"hit-image-bytes\n")

POST_ROW: dict[str, object] = {
    "id": 987654,
    "md5": HIT_MD5,
    "rating": "s",
    "source": "https://www.pixiv.net/artworks/123",
    "file_size": 123456,
    "image_width": 832,
    "image_height": 1216,
    "tag_string_character": "hakui_koyori unmapped_character",
    "tag_string_artist": "bob_artist",
    "tag_string_general": "long_hair smile",
    "tag_string_copyright": "some_series",
    "tag_string_meta": "highres",
}


def post_row(**overrides: object) -> dict[str, object]:
    return {**POST_ROW, **overrides}


def response(status: int, *, body: bytes = b"", headers: Mapping[str, str] | None = None) -> HttpResponse:
    return HttpResponse(status=status, headers=dict(headers or {}), body=body)


def hit_response(md5: str = HIT_MD5) -> HttpResponse:
    return response(200, body=json.dumps([post_row(md5=md5)]).encode("utf-8"))


class FakeFetch:
    """Queue of canned responses; records every requested URL."""

    def __init__(self, *responses: HttpResponse | Exception):
        self.responses = list(responses)
        self.urls: list[str] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        self.urls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected fetch: {url}")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class SleepLog:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def make_client(fetch: FakeFetch, **overrides: object) -> tuple[DanbooruClient, SleepLog]:
    sleeps = SleepLog()
    options: dict[str, object] = {"fetch": fetch, "min_interval": 0.0, "sleep": sleeps}
    options.update(overrides)
    return DanbooruClient(**options), sleeps  # type: ignore[arg-type]


# --------------------------------------------------------------------------- hashing


@pytest.mark.parametrize(
    ("data", "digest"),
    [
        (b"hello", "5d41402abc4b2a76b9719d911017c592"),
        (b"", "d41d8cd98f00b204e9800998ecf8427e"),
        (b"\xff\x00\x80raw\xfe bytes", "605fbec965aa557e634b5a8ddd5fe92e"),
    ],
    ids=["ascii", "empty", "raw-non-utf8"],
)
def test_file_md5_hashes_exact_file_bytes(tmp_path: Path, data: bytes, digest: str) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(data)

    assert file_md5(path) == digest


def test_file_md5_streaming_chunks_do_not_drop_tail(tmp_path: Path) -> None:
    data = bytes(range(256)) * 7 + b"tail"  # 1796 bytes: chunk boundaries land mid-payload
    path = tmp_path / "chunked.bin"
    path.write_bytes(data)

    for chunk_size in (1, 3, 1024, 1 << 20):
        assert file_md5(path, chunk_size=chunk_size) == "59d2a6e36c6ad5f909d456faa9e923c5"


def test_post_url_queries_by_md5_and_limits_one() -> None:
    client, _ = make_client(FakeFetch())
    url = client.post_url(HIT_MD5.upper())

    assert "tags=md5%3A" + HIT_MD5 in url
    parsed = urllib.parse.urlsplit(url)
    assert parsed.path == "/posts.json"
    assert dict(urllib.parse.parse_qsl(parsed.query)) == {"tags": f"md5:{HIT_MD5}", "limit": "1"}


# --------------------------------------------------------------------------- post parsing


def test_parse_post_types_every_field() -> None:
    post = parse_post(POST_ROW)

    assert post.site == SITE_DANBOORU
    assert post.post_id == 987654
    assert post.md5 == HIT_MD5
    assert (post.rating_raw, post.rating) == ("s", "sensitive")
    assert post.source == "https://www.pixiv.net/artworks/123"
    assert (post.file_size, post.width, post.height) == (123456, 832, 1216)
    assert post.character_tags == ("hakui_koyori", "unmapped_character")
    assert post.artist_tags == ("bob_artist",)
    assert post.general_tags == ("long_hair", "smile")
    assert post.copyright_tags == ("some_series",)
    assert post.meta_tags == ("highres",)
    assert post.has_character and post.has_artist and post.has_general
    assert post.raw == POST_ROW


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("g", "general"),
        ("G", "general"),
        ("s", "sensitive"),
        ("q", "questionable"),
        ("e", "explicit"),
        ("explicit", "explicit"),
        ("", None),
        ("x", None),
        (None, None),
    ],
)
def test_normalize_rating_maps_short_codes(raw: object, label: str | None) -> None:
    assert normalize_rating(raw) == label


def test_split_tag_string_normalizes_and_dedupes_preserving_first_spelling() -> None:
    assert split_tag_string("hakui_koyori  smile\nsmile") == ("hakui_koyori", "smile")
    assert split_tag_string(["Hakui_Koyori", "hakui koyori", "SMILE", "smile"]) == (
        "Hakui_Koyori",
        "SMILE",
    )
    assert split_tag_string(None) == ()


# --------------------------------------------------------------------------- vocabulary and mapping


def test_load_vocab_normalizes_keys_and_first_spelling_wins(tmp_path: Path) -> None:
    mapping_path = tmp_path / "vocab.json"
    mapping_path.write_text(
        json.dumps({"Hakui_Koyori": "hakui_koyori", "hakui koyori": "ignored", "": "blank"}),
        encoding="utf-8",
    )
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps(["Long Hair", "smile"]), encoding="utf-8")

    assert load_vocab(mapping_path) == {"hakui koyori": "hakui_koyori"}
    assert load_vocab(list_path) == {"long hair": "Long Hair", "smile": "smile"}


def test_map_post_tags_maps_vocab_drops_negatives_and_keeps_source_only() -> None:
    post = parse_post(
        post_row(
            tag_string_character="Hakui_Koyori Unmapped_Character",
            tag_string_general="Long_Hair bad_anatomy Watermark smile extra_tag",
        )
    )

    mapped = map_post_tags(
        post,
        {"hakui koyori": "hakui_koyori"},
        {"long hair": "long_hair"},
    )

    assert mapped.characters == ("hakui_koyori", "Unmapped_Character")
    assert mapped.new_character == ("Unmapped_Character",)
    assert mapped.tags == ("long_hair", "smile", "extra_tag")
    assert mapped.new_tags == ("smile", "extra_tag")


# --------------------------------------------------------------------------- HTTP lookup policy


def test_lookup_404_is_a_single_fetch_miss() -> None:
    fetch = FakeFetch(response(404))
    client, sleeps = make_client(fetch)

    result = client.lookup(HIT_MD5)

    assert (result.status, result.http_code, result.attempts) == (STATUS_MISS, 404, 1)
    assert result.post is None
    assert len(fetch.urls) == 1
    assert sleeps.calls == []


def test_lookup_empty_200_array_is_a_miss() -> None:
    fetch = FakeFetch(response(200, body=b"[]"))
    client, _ = make_client(fetch)

    result = client.lookup(HIT_MD5)

    assert (result.status, result.http_code, result.attempts) == (STATUS_MISS, 200, 1)
    assert result.post is None
    assert len(fetch.urls) == 1


def test_lookup_hit_parses_first_post() -> None:
    fetch = FakeFetch(hit_response())
    client, _ = make_client(fetch)

    result = client.lookup(HIT_MD5)

    assert result.hit and result.status == STATUS_HIT
    assert result.attempts == 1
    assert result.post is not None
    assert result.post.post_id == 987654
    assert result.post.rating == "sensitive"
    assert result.post.character_tags == ("hakui_koyori", "unmapped_character")


def test_lookup_429_retry_after_zero_then_hit() -> None:
    fetch = FakeFetch(response(429, headers={"Retry-After": "0"}), hit_response())
    client, sleeps = make_client(fetch)

    result = client.lookup(HIT_MD5)

    assert result.status == STATUS_HIT
    assert result.attempts == 2
    assert len(fetch.urls) == 2
    assert sleeps.calls == [0.0]


def test_lookup_429_retry_after_above_cap_is_non_terminal_error() -> None:
    fetch = FakeFetch(response(429, headers={"Retry-After": "301"}))
    client, sleeps = make_client(fetch)

    result = client.lookup(HIT_MD5)

    assert (result.status, result.http_code, result.attempts) == (STATUS_ERROR, 429, 1)
    assert result.status not in TERMINAL_STATUSES
    assert "exceeds cap" in (result.error or "")
    assert len(fetch.urls) == 1
    assert sleeps.calls == []


def test_lookup_500_then_hit_uses_capped_backoff() -> None:
    fetch = FakeFetch(response(500), hit_response())
    client, sleeps = make_client(fetch)

    result = client.lookup(HIT_MD5)

    assert result.status == STATUS_HIT
    assert result.attempts == 2
    assert len(fetch.urls) == 2
    assert sleeps.calls == [2.0]


@pytest.mark.parametrize("code", [401, 403])
def test_lookup_auth_block_latches_and_stops_fetching(code: int) -> None:
    fetch = FakeFetch(response(code))
    client, _ = make_client(fetch)

    first = client.lookup(HIT_MD5)
    second = client.lookup("0" * 32)

    assert (first.status, first.http_code, first.attempts) == (STATUS_AUTH_BLOCKED, code, 1)
    assert second.status == STATUS_AUTH_BLOCKED
    assert second.http_code == code
    assert second.attempts == 0
    assert client.blocked_code == code
    assert len(fetch.urls) == 1


# --------------------------------------------------------------------------- SQLite cache


def test_cache_creates_documented_tables(tmp_path: Path) -> None:
    path = tmp_path / "booru.sqlite"
    BooruCache(path).close()

    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    assert {"files", "site_probe", "posts", "meta"} <= tables


def test_cache_file_md5_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "booru.sqlite"
    sha = "a" * 64
    with BooruCache(path) as cache:
        assert cache.get_md5(sha) is None
        cache.record_file(sha, path="/pool/alice/aa/image.png", artist="alice", size=16, md5=HIT_MD5)
        assert cache.get_md5(sha) == HIT_MD5

    with BooruCache(path) as reopened:
        assert reopened.get_md5(sha) == HIT_MD5
        assert reopened.get_md5("b" * 64) is None


def test_cache_terminal_sha256_excludes_errors(tmp_path: Path) -> None:
    hit, miss, blocked, failed = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    with BooruCache(tmp_path / "booru.sqlite") as cache:
        for sha, status in (
            (hit, STATUS_HIT),
            (miss, STATUS_MISS),
            (blocked, STATUS_AUTH_BLOCKED),
            (failed, STATUS_ERROR),
        ):
            cache.record_probe(sha, site=SITE_DANBOORU, status=status)

        assert cache.terminal_sha256(SITE_DANBOORU) == {hit, miss, blocked}


def test_cache_auth_block_latch_survives_reopen_and_clears(tmp_path: Path) -> None:
    path = tmp_path / "booru.sqlite"
    with BooruCache(path) as cache:
        cache.mark_blocked(SITE_DANBOORU, 403)
        assert cache.blocked_code(SITE_DANBOORU) == 403

    with BooruCache(path) as reopened:
        assert reopened.blocked_code(SITE_DANBOORU) == 403
        reopened.clear_blocked(SITE_DANBOORU)

    with BooruCache(path) as cleared:
        assert cleared.blocked_code(SITE_DANBOORU) is None


def test_cache_post_round_trip_preserves_unicode(tmp_path: Path) -> None:
    raw = {"id": 987654, "tag_string_character": "ネコ hakui_koyori", "rating": "g"}
    with BooruCache(tmp_path / "booru.sqlite") as cache:
        cache.store_post(SITE_DANBOORU, 987654, json.dumps(raw, ensure_ascii=False))

        assert cache.load_post(SITE_DANBOORU, "987654") == raw
        assert cache.load_post(SITE_DANBOORU, 987654) == raw
        assert cache.load_post(SITE_DANBOORU, 1) is None
