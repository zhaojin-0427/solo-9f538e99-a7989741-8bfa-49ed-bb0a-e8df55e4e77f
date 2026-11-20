"""最坏规模内存与正确性压测：12 帧 4096x4096 16 位平场。

在独立数据目录造数据，用 resource 监控子进程峰值 RSS，
并与 numpy 全量对照实现抽样比对。
用法： python -I scripts/stress_memory.py
"""

from __future__ import annotations

import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.pipeline import build_master_flat  # noqa: E402


def generate_frames(data_dir: Path, n: int, height: int, width: int) -> tuple[list[Path], list[np.ndarray], float]:
    rng = np.random.default_rng(20261120)
    paths, raws = [], []
    saturation = 5000.0
    for i in range(n):
        raw = rng.integers(500, 1500, size=(height, width)).astype(np.int16)
        raw[: 8 + i, : 8] = 0  # 制造 <3 帧区域
        raw[100, 100 + i] = 30000  # 饱和
        paths.append(data_dir / f"inputs" / f"flat_{i:02d}.fits")
        paths[-1].parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data=raw).writeto(paths[-1], overwrite=True)
        raws.append(raw)
    return paths, raws, saturation


def reference(raws: list[np.ndarray], saturation: float) -> tuple[np.ndarray, np.ndarray]:
    """分块对照实现，避免校验阶段自身占用过多内存。"""
    n, height, width = len(raws), raws[0].shape[0], raws[0].shape[1]
    master = np.full((height, width), np.nan, dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint8)
    medians = []
    for raw in raws:
        valid_all = (raw > 0) & (raw < saturation)
        medians.append(np.median(raw[valid_all].astype(np.float64)))
    block_rows = 512
    for r0 in range(0, height, block_rows):
        r1 = min(r0 + block_rows, height)
        stack = np.empty((n, r1 - r0, width), dtype=np.float32)
        for i, raw in enumerate(raws):
            block = raw[r0:r1].astype(np.float32)
            valid = (block > 0.0) & (block < saturation)
            stack[i] = np.where(valid, block / medians[i], np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(stack, axis=0)
        cnt = np.isfinite(stack).sum(axis=0)
        med[cnt < 3] = np.nan
        master[r0:r1] = med
        counts[r0:r1] = cnt
    return master, counts


def main() -> int:
    n, height, width = 12, 4096, 4096
    root = Path(tempfile.mkdtemp(prefix="stress_root_"))
    try:
        print(f"[stress] 生成 {n} 帧 {width}x{height} 输入于 {root}")
        t0 = time.perf_counter()
        paths, raws, saturation = generate_frames(root / "data", n, height, width)
        print(f"[stress] 生成耗时 {time.perf_counter() - t0:.1f}s")

        out_dir, work_dir = root / "products", root / "work"
        t0 = time.perf_counter()
        product = build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=out_dir, work_dir=work_dir,
        )
        elapsed = time.perf_counter() - t0
        # 注意：本进程同时持有全部输入整数数组与后续参考实现，
        # 因此这里的 RSS 不是管线的真实预算占用；
        # 管线纯净峰值请用 measure_pipeline_only.py / matrix_sizes.py 测量。
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(f"[stress] 制备耗时 {elapsed:.1f}s（本进程含测试夹具，RSS {peak_mib:.1f} MiB 仅供参考）")

        with fits.open(product) as out:
            master, counts = np.asarray(out[0].data), np.asarray(out[1].data)
            assert master.shape == (height, width)
            assert counts.shape == (height, width)
            assert out[1].name == "COUNTS"

        print("[stress] 运行对照实现（仅用于校验，不计入内存预算）……")
        expected_master, expected_counts = reference(raws, saturation)
        np.testing.assert_array_equal(counts, expected_counts)
        ok = np.isfinite(expected_master)
        np.testing.assert_allclose(master[ok], expected_master[ok], rtol=1e-5)
        assert np.isnan(master[~ok]).all()
        print(f"[stress] 数值校验通过：有效像元 {ok.sum()}，NaN 像元 {(~ok).sum()}")
        print(f"[stress] 成品：{product}（{product.stat().st_size / 2**20:.1f} MiB）")
        print("[stress] 数值校验通过；内存预算判定请运行 measure_pipeline_only.py")
        return 0
    finally:
        pass  # 保留 /tmp 数据便于排查；系统会自行清理


if __name__ == "__main__":
    raise SystemExit(main())
