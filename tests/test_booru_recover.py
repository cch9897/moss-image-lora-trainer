"""Recovery pipeline tests: pool scan, JSONL contract, resume, auth-block stop, dry run."""

from __future__ import annotations

import json
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
    BooruCache,
    DanbooruClient,
    HttpResponse,
)
from scripts.booru_recover import (
    EXIT_ALREADY_BLOCKED,
    EXIT_BLOCKED,
    EXIT_OK,
    RecoveryConfig,
    iter_pool_images,
    main,
    run_recovery,
)

# Fixture bytes, their SHA-256 names and MD5 query keys.
HIT_DATA = b"hit-image-bytes\n"
HIT_SHA = "117642c8a513d2dd327f31e1eebf80e7335757dacb0a4d6b9f680497c52826b9"
HIT_MD5 = "fa9f36a9f75aaf65206848489317a1df"

MISS_DATA = b"miss-image-bytes\n"
MISS_SHA = "30dd7ab494163f900aee1618e9ae78d51a98536e5c2e8b15b7211f9e795b148a"
MISS_MD5 = "3ee0d5ce94684b78d790591ed7ce885c"

RETRY_DATA = b"retry-image-bytes\n"
RETRY_SHA = "ec5e482ac831bba1c00789fcd411e2242953adf1d60d96407b0121c64461ddca"
RETRY_MD5 = "5b8db5732992fcc03f6b4fed8ec67552"

BLOCKED = (
    (b"blocked-1\n", "d9cff20ec9114106568de23b62f5cef82c471dbe10224f48da5c46288155a215"),
    (b"blocked-2\n", "ed0dd2f46c16aaf65719c3b560cb4dce3a529acd956544894a846061e66c04d7"),
    (b"blocked-3\n", "c09236eae82e0c72e9c90fde6b100189a36c1e3949b22f6691f4196a61487ffc"),
)

DRY_IMAGES = (
    (b"dry-1\n", "3c5649070ebcdd00f34bd84275edb9d6b43db785946e730f5148a2109da05d72"),
    (b"dry-2\n", "edd9421c65e7373e12109a624a5e606f6ce9c18435d83efd7da791b0768dbfde"),
)

ROW_KEYS = {
    "sha256",
    "site",
    "artist",
    "characters",
    "tags",
    "rating",
    "rating_label",
    "source",
    "source_host",
    "post_id",
    "hit",
    "status",
    "md5",
    "api_md5",
    "file_size",
    "width",
    "height",
    "booru_artist",
    "artist_match",
    "new_character",
    "new_tags",
    "character_count",
    "general_count",
    "http_code",
    "error",
}


def hit_row(md5: str) -> dict[str, object]:
    return {
        "id": 987654,
        "md5": md5,
        "rating": "s",
        "source": "https://www.pixiv.net/artworks/123",
        "file_size": 123456,
        "image_width": 832,
        "image_height": 1216,
        "tag_string_character": "hakui_koyori zz_source_only_char",
        "tag_string_artist": "bob_artist",
        "tag_string_general": "long_hair zz_source_only_general",
        "tag_string_copyright": "zz_copyright",
        "tag_string_meta": "zz_meta",
    }


FORBIDDEN_IN_REPORT = (
    "hakui_koyori",
    "zz_source_only_char",
    "zz_source_only_general",
    "bob_artist",
    "zz_copyright",
    "zz_meta",
)


class FakeFetch:
    """Respond by the ``md5:`` query value, optionally with queued responses."""

    def __init__(
        self,
        responses: Mapping[str, HttpResponse | list[HttpResponse]] | None = None,
        *,
        default: HttpResponse | None = None,
    ):
        self.responses = {key: list(value) if isinstance(value, list) else value
                          for key, value in (responses or {}).items()}
        self.default = default
        self.urls: list[str] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        self.urls.append(url)
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        digest = query["tags"].removeprefix("md5:")
        item = self.responses.get(digest, self.default)
        if isinstance(item, list):
            if not item:
                raise AssertionError(f"no queued response for {digest}")
            item = item.pop(0)
        if item is None:
            raise AssertionError(f"unexpected fetch for {digest}")
        return item


def json_response(rows: list[Mapping[str, object]], status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, headers={"Content-Type": "application/json"},
                        body=json.dumps(rows).encode("utf-8"))


def client_for(fetch: FakeFetch, **overrides: object) -> DanbooruClient:
    options: dict[str, object] = {"fetch": fetch, "min_interval": 0.0, "sleep": lambda seconds: None}
    options.update(overrides)
    return DanbooruClient(**options)  # type: ignore[arg-type]


def write_pool_image(
    root: Path,
    artist: str,
    sha256: str,
    data: bytes,
    *,
    shard: str = "aa",
    ext: str = ".png",
) -> Path:
    path = root / artist / shard / f"{sha256}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def recovery_config(tmp_path: Path, **overrides: object) -> RecoveryConfig:
    options: dict[str, object] = {
        "pool_root": tmp_path / "pool",
        "cache_path": tmp_path / "state" / "cache.sqlite",
        "out_path": tmp_path / "state" / "out.jsonl",
        "report_path": tmp_path / "state" / "report.md",
        "vocab_dir": tmp_path / "vocab",
        "quiet": True,
        "min_interval": 0.0,
    }
    options.update(overrides)
    return RecoveryConfig(**options)  # type: ignore[arg-type]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_vocabs(tmp_path: Path) -> Path:
    vocab_dir = tmp_path / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    (vocab_dir / "character_vocab.json").write_text(
        json.dumps({"hakui koyori": "hakui_koyori"}), encoding="utf-8"
    )
    (vocab_dir / "general_vocab.json").write_text(
        json.dumps({"long hair": "long_hair"}), encoding="utf-8"
    )
    return vocab_dir


# --------------------------------------------------------------------------- pool scan


def test_iter_pool_images_selects_only_still_originals(tmp_path: Path) -> None:
    root = tmp_path / "images"
    keep = [
        write_pool_image(root, "alice", HIT_SHA, HIT_DATA, shard="aa", ext=".jpg"),
        write_pool_image(root, "alice", MISS_SHA, MISS_DATA, shard="ab", ext=".PNG"),
        write_pool_image(root, "alice", DRY_IMAGES[0][1], DRY_IMAGES[0][0], shard="ac", ext=".jpeg"),
        write_pool_image(root, "bob", RETRY_SHA, RETRY_DATA, shard="cd", ext=".webp"),
    ]
    junk = root / "alice" / "ac"
    (junk / f"{'e' * 64}.gif").write_bytes(b"gif")
    (junk / f"{'f' * 64}.mp4").write_bytes(b"mp4")
    (junk / f"{'0' * 64}.txt").write_text("text", encoding="utf-8")
    (junk / "not-a-hash.png").write_bytes(b"png")
    (junk / f"{DRY_IMAGES[1][1]}.jpg").mkdir()  # a directory is not an image
    (root / "alice" / "not-a-shard").mkdir()
    (root / "alice" / "not-a-shard" / f"{'1' * 64}.png").write_bytes(b"png")
    (root / ".thumbs").mkdir()
    (root / ".thumbs" / f"{'2' * 64}.jpg").write_bytes(b"jpg")
    (root / "alice" / ".thumbs").mkdir()
    (root / "alice" / ".thumbs" / f"{'3' * 64}.jpg").write_bytes(b"jpg")
    (root / "loose.png").write_bytes(b"png")

    found = list(iter_pool_images(root))

    assert [(image.sha256, image.artist, image.path.name, image.size) for image in found] == [
        (HIT_SHA, "alice", f"{HIT_SHA}.jpg", len(HIT_DATA)),
        (MISS_SHA, "alice", f"{MISS_SHA}.PNG", len(MISS_DATA)),
        (DRY_IMAGES[0][1], "alice", f"{DRY_IMAGES[0][1]}.jpeg", len(DRY_IMAGES[0][0])),
        (RETRY_SHA, "bob", f"{RETRY_SHA}.webp", len(RETRY_DATA)),
    ]
    assert {image.path for image in found} == set(keep)


# --------------------------------------------------------------------------- end-to-end


def test_run_recovery_writes_hit_and_miss_rows_without_leaking_tags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pool = tmp_path / "pool"
    write_pool_image(pool, "alice", HIT_SHA, HIT_DATA, shard="ab")
    write_pool_image(pool, "zoe", MISS_SHA, MISS_DATA, shard="cd")
    config = recovery_config(tmp_path, vocab_dir=write_vocabs(tmp_path), quiet=False)
    fetch = FakeFetch({
        HIT_MD5: json_response([hit_row(HIT_MD5)]),
        MISS_MD5: HttpResponse(status=404),
    })

    run = run_recovery(config, client=client_for(fetch))

    assert (run["eligible"], run["selected"], run["processed"]) == (2, 2, 2)
    assert (run["hits"], run["misses"], run["errors"]) == (1, 1, 0)
    assert run["exit_code"] == EXIT_OK
    assert [(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["tags"][0]) for url in fetch.urls] == [
        f"md5:{HIT_MD5}",
        f"md5:{MISS_MD5}",
    ]

    rows = {row["sha256"]: row for row in read_jsonl(config.out_path)}
    assert set(rows) == {HIT_SHA, MISS_SHA}

    hit = rows[HIT_SHA]
    assert set(hit) == ROW_KEYS
    assert (hit["site"], hit["status"], hit["hit"], hit["http_code"]) == (
        SITE_DANBOORU,
        STATUS_HIT,
        True,
        200,
    )
    assert (hit["artist"], hit["booru_artist"], hit["artist_match"]) == ("alice", ["bob_artist"], False)
    assert hit["characters"] == ["hakui_koyori", "zz_source_only_char"]
    assert hit["tags"] == ["long_hair", "zz_source_only_general"]
    assert hit["new_character"] == ["zz_source_only_char"]
    assert hit["new_tags"] == ["zz_source_only_general"]
    assert (hit["character_count"], hit["general_count"]) == (2, 2)
    assert (hit["rating"], hit["rating_label"]) == ("s", "sensitive")
    assert (hit["source"], hit["source_host"]) == ("https://www.pixiv.net/artworks/123", "www.pixiv.net")
    assert (hit["post_id"], hit["md5"], hit["api_md5"]) == (987654, HIT_MD5, HIT_MD5)
    assert (hit["file_size"], hit["width"], hit["height"]) == (123456, 832, 1216)
    assert hit["error"] is None

    miss = rows[MISS_SHA]
    assert (miss["artist"], miss["status"], miss["hit"], miss["http_code"]) == (
        "zoe",
        STATUS_MISS,
        False,
        404,
    )
    assert (miss["characters"], miss["tags"], miss["booru_artist"], miss["new_character"]) == ([], [], [], [])
    assert (miss["rating"], miss["rating_label"], miss["post_id"]) == (None, None, None)
    assert miss["md5"] == MISS_MD5

    with BooruCache(config.cache_path) as cache:
        stats = cache.stats(SITE_DANBOORU)
    assert (stats["files"], stats["files_with_md5"], stats["posts"]) == (2, 2, 1)
    assert (stats["hits"], stats["misses"], stats["errors"]) == (1, 1, 0)
    assert stats["ratings"] == {"s": 1}
    assert stats["source_hosts"] == {"www.pixiv.net": 1}

    report = config.report_path.read_text(encoding="utf-8")
    assert "| Posts probed | 2 |" in report
    assert "| Hits | 1 |" in report
    assert "| Misses | 1 |" in report
    assert "www.pixiv.net" in report
    captured = capsys.readouterr()
    for fragment in FORBIDDEN_IN_REPORT:
        assert fragment not in report
        assert fragment not in captured.err
        assert fragment not in captured.out


def test_resume_rerun_makes_zero_fetches_and_appends_no_rows(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    write_pool_image(pool, "alice", HIT_SHA, HIT_DATA, shard="ab")
    write_pool_image(pool, "zoe", MISS_SHA, MISS_DATA, shard="cd")
    config = recovery_config(tmp_path, vocab_dir=write_vocabs(tmp_path))
    first = FakeFetch({
        HIT_MD5: json_response([hit_row(HIT_MD5)]),
        MISS_MD5: HttpResponse(status=404),
    })
    run_recovery(config, client=client_for(first))
    lines_before = config.out_path.read_text(encoding="utf-8").splitlines()

    second = FakeFetch()
    rerun = run_recovery(config, client=client_for(second))

    assert second.urls == []
    assert (rerun["processed"], rerun["skipped"]) == (0, 2)
    assert rerun["exit_reason"] == "nothing to do (all selected rows terminal)"
    assert config.out_path.read_text(encoding="utf-8").splitlines() == lines_before
    assert len(lines_before) == 2


def test_error_rows_are_cached_but_not_appended_and_retry_on_resume(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    write_pool_image(pool, "alice", RETRY_SHA, RETRY_DATA)
    config = recovery_config(tmp_path, vocab_dir=write_vocabs(tmp_path))
    failing = FakeFetch({RETRY_MD5: HttpResponse(status=500)})

    failed = run_recovery(config, client=client_for(failing, max_attempts=1))

    assert (failed["processed"], failed["errors"], failed["hits"]) == (0, 1, 0)
    assert read_jsonl(config.out_path) == []
    with BooruCache(config.cache_path) as cache:
        probe = cache.probe(RETRY_SHA, SITE_DANBOORU)
    assert probe is not None and probe["status"] == STATUS_ERROR
    assert RETRY_SHA not in BooruCache(config.cache_path).terminal_sha256(SITE_DANBOORU)

    succeeding = FakeFetch({RETRY_MD5: json_response([hit_row(RETRY_MD5)])})
    retried = run_recovery(config, client=client_for(succeeding))

    assert retried["hits"] == 1
    assert len(succeeding.urls) == 1
    rows = read_jsonl(config.out_path)
    assert [row["sha256"] for row in rows] == [RETRY_SHA]
    assert (rows[0]["status"], rows[0]["hit"]) == (STATUS_HIT, True)


def test_auth_blocked_stops_immediately_and_persists_across_runs(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    for index, (data, sha) in enumerate(BLOCKED):
        write_pool_image(pool, "alice", sha, data, shard=f"a{index}")
    config = recovery_config(tmp_path, vocab_dir=write_vocabs(tmp_path))
    blocked = FakeFetch(default=HttpResponse(status=403))

    run = run_recovery(config, client=client_for(blocked))

    assert len(blocked.urls) == 1
    assert run["exit_code"] == EXIT_BLOCKED
    assert run["exit_reason"] == "auth_blocked (HTTP 403)"
    assert run["processed"] == 1
    rows = read_jsonl(config.out_path)
    assert [(row["sha256"], row["status"]) for row in rows] == [(BLOCKED[0][1], STATUS_AUTH_BLOCKED)]

    with BooruCache(config.cache_path) as cache:
        assert cache.blocked_code(SITE_DANBOORU) == 403
        assert cache.probe(BLOCKED[0][1], SITE_DANBOORU)["status"] == STATUS_AUTH_BLOCKED
        assert cache.probe(BLOCKED[1][1], SITE_DANBOORU) is None
        assert cache.probe(BLOCKED[2][1], SITE_DANBOORU) is None

    later = FakeFetch(default=HttpResponse(status=403))
    rerun = run_recovery(config, client=client_for(later))

    assert rerun["exit_code"] == EXIT_ALREADY_BLOCKED
    assert later.urls == []


def test_dry_run_reports_counts_without_writing_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pool = tmp_path / "pool"
    write_pool_image(pool, "alice", DRY_IMAGES[0][1], DRY_IMAGES[0][0], shard="ab", ext=".jpg")
    write_pool_image(pool, "zoe", DRY_IMAGES[1][1], DRY_IMAGES[1][0], shard="cd", ext=".webp")
    cache = tmp_path / "state" / "cache.sqlite"
    out = tmp_path / "state" / "out.jsonl"
    report = tmp_path / "state" / "report.md"

    code = main([
        "--pool", str(pool),
        "--cache", str(cache),
        "--out", str(out),
        "--report", str(report),
        "--vocab-dir", str(tmp_path / "vocab"),
        "--dry-run",
        "--quiet",
    ])

    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert (payload["eligible"], payload["selected"], payload["exit_code"]) == (2, 2, EXIT_OK)
    assert payload["exit_reason"] == "dry-run"
    assert payload["by_extension"] == {".jpg": 1, ".webp": 1}
    assert payload["bytes"] == sum(len(data) for data, _ in DRY_IMAGES)
    assert not cache.exists()
    assert not out.exists()
    assert not report.exists()
