"""Smoke test — boots the FastAPI app in --no-model mode and exercises endpoints.

Run:
    pytest sam3_annotator/tests/

Requires sam3_annotator installed and a path containing at least one episode dir
with `{EP}_snippets.json`. Set SAM3_ANNOT_TEST_DATA_DIR or the test will skip.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


DATA_DIR = os.environ.get("SAM3_ANNOT_TEST_DATA_DIR")


@pytest.fixture(scope="session")
def client():
    if not DATA_DIR or not Path(DATA_DIR).is_dir():
        pytest.skip("SAM3_ANNOT_TEST_DATA_DIR not set or not a directory")
    os.environ["SAM3_ANNOT_DATA_DIR"] = DATA_DIR
    os.environ["SAM3_ANNOT_NO_MODEL"] = "1"
    os.environ["SAM3_ANNOT_NO_TEXT"] = "1"
    from fastapi.testclient import TestClient
    from sam3_annotator.server.app import app
    with TestClient(app) as c:
        yield c


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "SAM3 Annotator" in r.text


def test_version(client):
    r = client.get("/api/version")
    assert r.status_code == 200
    j = r.json()
    assert j["name"] == "sam3_annotator"
    assert "version" in j


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["no_model"] is True
    assert "data_dir" in j


def test_episodes(client):
    r = client.get("/api/episodes")
    assert r.status_code == 200
    eps = r.json().get("episodes", [])
    assert isinstance(eps, list)


def test_snippet_list(client):
    eps = client.get("/api/episodes").json()["episodes"]
    if not eps:
        pytest.skip("no episodes in test data dir")
    r = client.get(f"/api/episodes/{eps[0]}/snippets")
    assert r.status_code == 200
    assert "snippets" in r.json()


def test_session_open_close(client):
    eps = client.get("/api/episodes").json()["episodes"]
    if not eps:
        pytest.skip("no episodes in test data dir")
    snips = client.get(f"/api/episodes/{eps[0]}/snippets").json()["snippets"]
    if not snips:
        pytest.skip("no snippets")
    sid = snips[0]["snippet_id"]
    r = client.post("/api/session/open", json={
        "episode": eps[0], "snippet_id": sid,
        "categories": ["Tool", "Liver", "Gallbladder"],
    })
    assert r.status_code == 200
    j = r.json()
    assert j["episode"] == eps[0]
    assert j["snippet_id"] == sid

    # Frame fetch
    r = client.get("/api/frame/0")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")

    # Masks for current frame
    r = client.get("/api/masks/0")
    assert r.status_code == 200
    assert "approved" in r.json()

    # Master mask list
    r = client.get("/api/masks/list")
    assert r.status_code == 200
    j = r.json()
    assert "items" in j
    assert "totals" in j

    # Close
    r = client.post("/api/session/close")
    assert r.status_code == 200
