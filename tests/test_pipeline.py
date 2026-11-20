"""管线层测试：数值正确性、过滤规则与各类整批失败。"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from app.pipeline import (
    MAX_FRAMES,
    MIN_FRAMES,
    PreparationError,
    build_master_flat,
    _median_scaled_from_hist,
)
from conftest import make_flat_fits, reference_master


def test_master_matches_reference(tmp_path, batch_factory):
    paths, raws, saturation = batch_factory(n=5)
    product = build_master_flat(
        paths, [p.name for p in paths], saturation,
        output_dir=tmp_path / "out", work_dir=tmp_path / "work",
    )
    expected_master, expected_counts = reference_master(
        raws, 1.0, 0.0, saturation, blank=None
    )
    with fits.open(product) as out:
        assert len(out) == 2
        master, counts = out[0].data, out[1].data
        assert master.dtype.kind == "f" and master.dtype.itemsize == 4
        assert counts.dtype == np.uint8
        np.testing.assert_array_equal(counts, expected_counts)
        ok = np.isfinite(expected_master)
        np.testing.assert_allclose(master[ok], expected_master[ok], rtol=1e-5)
        assert np.isnan(master[~ok]).all()
        assert out[0].header["NFRAMES"] == 5
        assert out[1].name == "COUNTS"


def test_blank_bscale_bzero_pipeline(tmp_path, batch_factory):
    blank = -32768
    paths, raws, saturation = batch_factory(
        n=6, bscale=0.5, bzero=10.0, blank=blank
    )
    product = build_master_flat(
        paths, [p.name for p in paths], saturation,
        output_dir=tmp_path / "out", work_dir=tmp_path / "work",
    )
    expected_master, expected_counts = reference_master(
        raws, 0.5, 10.0, saturation, blank=blank
    )
    with fits.open(product) as out:
        master, counts = out[0].data, out[1].data
        np.testing.assert_array_equal(counts, expected_counts)
        ok = np.isfinite(expected_master)
        np.testing.assert_allclose(master[ok], expected_master[ok], rtol=1e-4)
        assert np.isnan(master[~ok]).all()


def test_pixels_with_fewer_than_three_frames_are_nan(tmp_path):
    rng = np.random.default_rng(0)
    raws = [rng.integers(500, 800, size=(20, 20)).astype(np.int16) for _ in range(4)]
    # 同一位置只在两帧里有效：另外两帧填 0。
    raws[0][5, 5] = 0
    raws[1][5, 5] = 0
    paths = [
        make_flat_fits(tmp_path / f"f{i}.fits", raw) for i, raw in enumerate(raws)
    ]
    product = build_master_flat(
        paths, [p.name for p in paths], 5000.0,
        output_dir=tmp_path / "out", work_dir=tmp_path / "work",
    )
    with fits.open(product) as out:
        assert np.isnan(out[0].data[5, 5])
        assert out[1].data[5, 5] == 2
        assert np.isfinite(out[0].data[0, 0])
        assert out[1].data[0, 0] == 4


def test_saturation_excludes_pixels(tmp_path):
    rng = np.random.default_rng(1)
    raws = [rng.integers(500, 800, size=(16, 16)).astype(np.int16) for _ in range(4)]
    for raw in raws:
        raw[2, 2] = 9000
    paths = [make_flat_fits(tmp_path / f"s{i}.fits", raw) for i, raw in enumerate(raws)]
    product = build_master_flat(
        paths, [p.name for p in paths], 4000.0,
        output_dir=tmp_path / "out", work_dir=tmp_path / "work",
    )
    with fits.open(product) as out:
        assert np.isnan(out[0].data[2, 2])
        assert out[1].data[2, 2] == 0


def test_histogram_median_exact_for_even_and_odd_counts():
    rng = np.random.default_rng(7)
    for total in (1, 2, 3, 4, 100, 101):
        values = rng.integers(-1000, 1000, size=total)
        hist = np.bincount(values + 32768, minlength=65536)
        bscale, bzero = 2.0, -5.0
        got = _median_scaled_from_hist(hist, total, bscale, bzero)
        want = np.median(values.astype(np.float64) * bscale + bzero)
        assert got == pytest.approx(want)


def test_histogram_median_negative_bscale():
    values = np.array([-3, 0, 2, 2, 9], dtype=np.int32)
    hist = np.bincount(values + 32768, minlength=65536)
    bscale = -1.5
    got = _median_scaled_from_hist(hist, len(values), bscale, 7.0)
    want = np.median(values.astype(np.float64) * bscale + 7.0)
    assert got == pytest.approx(want)


def test_frame_without_valid_pixels_fails(tmp_path):
    raw = np.full((16, 16), 30000, dtype=np.int16)
    paths = [make_flat_fits(tmp_path / f"dead{i}.fits", raw) for i in range(3)]
    with pytest.raises(PreparationError, match="不存在任何有效像元"):
        build_master_flat(
            paths, [p.name for p in paths], 5000.0,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_truncated_file_fails_batch(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    data = paths[1].read_bytes()
    paths[1].write_bytes(data[: len(data) // 2])
    with pytest.raises(PreparationError, match="截断|无法解析"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )
    out_dir = tmp_path / "out"
    assert not out_dir.exists() or not list(out_dir.iterdir())


def test_non_fits_content_fails(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    paths[0].write_bytes(b"not a fits file at all" * 32)
    with pytest.raises(PreparationError):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )


def test_wrong_bitpix_fails(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    data = fits.open(paths[0])[0].data.astype(np.int32)
    fits.PrimaryHDU(data=data).writeto(paths[0], overwrite=True)
    with pytest.raises(PreparationError, match="BITPIX"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )


def test_compressed_hdu_fails(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    raw = fits.open(paths[0])[0].data
    comp = fits.CompImageHDU(data=np.asarray(raw, dtype=np.int16))
    fits.HDUList([fits.PrimaryHDU(), comp]).writeto(paths[0], overwrite=True)
    with pytest.raises(PreparationError, match="压缩"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )


def test_shape_mismatch_fails(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    other = np.ones((32, 32), dtype=np.int16) * 600
    make_flat_fits(paths[2], other)
    with pytest.raises(PreparationError, match="尺寸"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )


def test_filter_mismatch_fails(tmp_path, batch_factory):
    from astropy.io import fits as afits

    paths, _, saturation = batch_factory(n=3)
    with afits.open(paths[1], mode="update") as hdul:
        hdul[0].header["FILTER"] = "R"
    with afits.open(paths[0]) as hdul:
        hdul[0].header.setdefault("FILTER", "V")
    # 首帧补一个不同滤镜，制造 V/R 冲突。
    with afits.open(paths[0], mode="update") as hdul:
        hdul[0].header["FILTER"] = "V"
    with pytest.raises(PreparationError, match="滤镜|相机"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )


def test_instrument_mismatch_fails(tmp_path, batch_factory):
    from astropy.io import fits as afits

    paths, _, saturation = batch_factory(n=3)
    for i, p in enumerate(paths):
        with afits.open(p, mode="update") as hdul:
            hdul[0].header["INSTRUME"] = "CAM-A" if i % 2 == 0 else "CAM-B"
    with pytest.raises(PreparationError, match="相机|滤镜"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )


def test_matching_camera_filter_passes(tmp_path, batch_factory):
    from astropy.io import fits as afits

    paths, _, saturation = batch_factory(n=3)
    for p in paths:
        with afits.open(p, mode="update") as hdul:
            hdul[0].header["INSTRUME"] = "CAM-X"
            hdul[0].header["FILTER"] = "g"
    product = build_master_flat(
        paths, [p.name for p in paths], saturation,
        output_dir=tmp_path / "out", work_dir=tmp_path / "work",
    )
    with afits.open(product) as out:
        assert out[0].header["IMAGTYPE"] == "MASTERFLAT"


def test_two_image_hdus_fails(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    raw = fits.open(paths[0])[0].data.astype(np.int16)
    hdul = fits.HDUList(
        [fits.PrimaryHDU(data=raw), fits.ImageHDU(data=raw, name="EXTRA")]
    )
    hdul.writeto(paths[0], overwrite=True)
    with pytest.raises(PreparationError, match="幅图像"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )


def test_frame_count_limits(tmp_path):
    raw = np.ones((8, 8), dtype=np.int16) * 600
    paths = [make_flat_fits(tmp_path / f"x{i}.fits", raw) for i in range(MIN_FRAMES - 1)]
    with pytest.raises(PreparationError, match="帧数"):
        build_master_flat(
            paths, [p.name for p in paths], 5000.0,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )

    paths = [make_flat_fits(tmp_path / f"y{i}.fits", raw) for i in range(MAX_FRAMES + 1)]
    with pytest.raises(PreparationError, match="帧数"):
        build_master_flat(
            paths, [p.name for p in paths], 5000.0,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )


@pytest.mark.parametrize("bad_saturation", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_saturation(tmp_path, batch_factory, bad_saturation):
    paths, _, _ = batch_factory(n=3)
    with pytest.raises(PreparationError, match="饱和阈值"):
        build_master_flat(
            paths, [p.name for p in paths], bad_saturation,
            output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        )


def test_write_failure_reports_batch_error(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    # 输出目录内部被同名普通文件阻塞，writeto 必然失败；不得留下半成品。
    blocked_output = tmp_path / "blocked"
    blocked_output.mkdir()
    (blocked_output / "child").write_text("not a directory")
    with pytest.raises(PreparationError) as exc_info:
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=blocked_output / "child" / "out",
            work_dir=work_dir,
        )
    assert exc_info.value.status_code == 500
    assert not list(work_dir.rglob("*.tmp"))
