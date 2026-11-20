"""测试用 FITS 夹具：全部在临时目录中现造，与测试脚本目录隔离。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def make_flat_fits(
    path: Path,
    data: np.ndarray,
    bscale: float = 1.0,
    bzero: float = 0.0,
    blank: int | None = None,
    name: str | None = None,
) -> Path:
    """写一幅 16 位整数二维主图 FITS（原始整数 + BSCALE/BZERO 头）。"""
    raw = np.asarray(data, dtype=np.int16)
    hdu = fits.PrimaryHDU(data=raw)
    hdu.header["BSCALE"] = bscale
    hdu.header["BZERO"] = bzero
    if blank is not None:
        hdu.header["BLANK"] = blank
    if name is not None:
        hdu.header["OBJECT"] = name
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([hdu]).writeto(path, overwrite=True)
    return path


def reference_master(
    raws: list[np.ndarray],
    bscale: float,
    bzero: float,
    saturation: float,
    blank: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """直接用 numpy 全量数组实现一遍规约，作为管线的对照实现。"""
    normed = []
    for raw in raws:
        block = raw.astype(np.int32)
        value = block.astype(np.float64) * bscale + bzero
        valid = (value > 0.0) & (value < saturation)
        if blank is not None:
            valid &= block != blank
        med = np.median(value[valid])
        normed.append(np.where(valid, value / med, np.nan))
    stack = np.stack(normed, axis=0)
    with np.errstate(all="ignore"):
        master = np.nanmedian(stack, axis=0)
    counts = np.isfinite(stack).sum(axis=0)
    master[counts < 3] = np.nan
    return master.astype(np.float32), counts.astype(np.uint8)


@pytest.fixture
def batch_factory(tmp_path):
    """造一批同尺寸平场，可指定每帧有效掩码的差异（制造 <3 帧位置）。"""

    def _factory(
        n: int = 5,
        height: int = 64,
        width: int = 48,
        seed: int = 42,
        bscale: float = 1.0,
        bzero: float = 0.0,
        blank: int | None = None,
    ) -> tuple[list[Path], list[np.ndarray], float]:
        rng = np.random.default_rng(seed)
        paths: list[Path] = []
        raws: list[np.ndarray] = []
        # 阈值与换算单位一致：raw 主力值约 400~900，注入的饱和值在 30000。
        saturation = 5000.0
        for i in range(n):
            base = rng.integers(400, 900, size=(height, width), dtype=np.int32)
            noise = rng.integers(-30, 30, size=(height, width), dtype=np.int32)
            raw = base + noise
            # 每帧挖掉不同角落，使部分位置仅 1~2 帧有效。
            raw[: 4 + i, : 4 + i] = -1
            # 注入饱和像元与非正像元。
            raw[20 + i, 20] = 30000
            raw[30, 30 + i] = 0
            if blank is not None:
                raw[40, 40 + (i % 3)] = blank
            raw = np.clip(raw, -30000, 30000).astype(np.int16)
            raws.append(raw)
            paths.append(
                make_flat_fits(
                    tmp_path / f"frame_{i:02d}.fits",
                    raw,
                    bscale=bscale,
                    bzero=bzero,
                    blank=blank,
                )
            )
        return paths, raws, saturation

    return _factory
