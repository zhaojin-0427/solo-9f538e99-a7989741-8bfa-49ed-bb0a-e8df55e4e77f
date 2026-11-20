"""/prepare 与 /docs 接口测试。"""

from __future__ import annotations

import io

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from app import main
from conftest import make_flat_fits


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "products")
    (tmp_path / "work").mkdir()
    (tmp_path / "products").mkdir()
    with TestClient(main.app) as c:
        yield c


def _upload(client, paths, saturation=5000.0):
    files = [
        ("files", (p.name, p.read_bytes(), "application/fits")) for p in paths
    ]
    return client.post(
        "/prepare", data={"saturation": str(saturation)}, files=files
    )


def test_health_and_docs(client):
    assert client.get("/health").json() == {"status": "ok"}
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "主平场" in docs.text
    openapi = client.get("/openapi.json").json()
    assert "/prepare" in openapi["paths"]


def test_prepare_happy_path(client, tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=4)
    resp = _upload(client, paths, saturation)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/fits"
    assert resp.headers["x-frame-count"] == "4"
    with fits.open(io.BytesIO(resp.content)) as out:
        assert out[0].header["IMAGTYPE"] == "MASTERFLAT"
        assert out[0].header["SATURATE"] == saturation
        assert out[1].name == "COUNTS"
        assert out[1].data.shape == out[0].data.shape
    # 成品已落盘且临时上传目录被清理。
    products = list((tmp_path / "products").glob("master_flat_*.fits"))
    assert len(products) == 1
    assert not any((tmp_path / "work").glob("batch_*"))


def test_too_few_files_rejected(client, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    resp = _upload(client, paths[:2], saturation)
    assert resp.status_code == 422
    assert "帧数" in resp.json()["detail"]


def test_too_many_files_rejected(client, tmp_path):
    raw = np.ones((8, 8), dtype=np.int16) * 600
    paths = [make_flat_fits(tmp_path / f"m{i}.fits", raw) for i in range(13)]
    resp = _upload(client, paths)
    assert resp.status_code == 422
    assert "帧数" in resp.json()["detail"]


def test_bad_saturation_form_value(client, batch_factory):
    paths, _, _ = batch_factory(n=3)
    files = [("files", (p.name, p.read_bytes(), "application/fits")) for p in paths]
    resp = client.post(
        "/prepare", data={"saturation": "-3"}, files=files
    )
    assert resp.status_code == 422
    assert "饱和阈值" in resp.json()["detail"]


def test_truncated_upload_returns_422_and_no_product(client, tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    blob = paths[1].read_bytes()
    files = [
        ("files", (p.name, p.read_bytes() if i != 1 else blob[:40], "application/fits"))
        for i, p in enumerate(paths)
    ]
    resp = client.post("/prepare", data={"saturation": str(saturation)}, files=files)
    assert resp.status_code == 422
    assert list((tmp_path / "products").iterdir()) == []


def test_all_dead_frames_returns_422(client, tmp_path):
    raw = np.full((12, 12), 30000, dtype=np.int16)
    paths = [make_flat_fits(tmp_path / f"d{i}.fits", raw) for i in range(3)]
    resp = _upload(client, paths, saturation=5000.0)
    assert resp.status_code == 422
    assert "有效像元" in resp.json()["detail"]


def test_missing_saturation_field_is_422(client, batch_factory):
    paths, _, _ = batch_factory(n=3)
    files = [("files", (p.name, p.read_bytes(), "application/fits")) for p in paths]
    resp = client.post("/prepare", files=files)
    assert resp.status_code == 422


def test_non_numeric_saturation_is_422(client, batch_factory):
    paths, _, _ = batch_factory(n=3)
    files = [("files", (p.name, p.read_bytes(), "application/fits")) for p in paths]
    resp = client.post("/prepare", data={"saturation": "abc"}, files=files)
    assert resp.status_code == 422
