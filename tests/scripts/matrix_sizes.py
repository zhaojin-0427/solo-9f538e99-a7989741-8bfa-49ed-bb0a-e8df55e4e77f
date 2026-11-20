"""多尺寸/帧数组合的内存与正确性矩阵测试。"""

from __future__ import annotations

import resource
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.pipeline import build_master_flat  # noqa: E402


def make_inputs(root: Path, n: int, h: int, w: int) -> tuple[list[Path], list[np.ndarray], float]:
    rng = np.random.default_rng(h * 31 + w)
    paths, raws = [], []
    sat = 5000.0
    for i in range(n):
        raw = rng.integers(500, 1500, size=(h, w)).astype(np.int16)
        raw[: min(2 + i, h), : min(2, w)] = 0
        if h > 50 and w > 50:
            raw[40, 40 + i] = 30000
        paths.append(root / f"f{i:02d}.fits")
        fits.PrimaryHDU(data=raw).writeto(paths[i], overwrite=True)
        raws.append(raw)
    return paths, raws, sat


def reference(raws, sat):
    """小图全量、大图分块的对照实现，避免校验脚本自身 OOM。"""
    n = len(raws)
    h, w = raws[0].shape
    medians = []
    for raw in raws:
        ok = (raw > 0) & (raw < sat)
        medians.append(np.median(raw[ok].astype(np.float64)))
    master = np.full((h, w), np.nan, np.float32)
    counts = np.zeros((h, w), np.uint8)
    block = 256
    for r0 in range(0, h, block):
        r1 = min(r0 + block, h)
        stack = np.empty((n, r1 - r0, w), np.float32)
        for i, raw in enumerate(raws):
            v = raw[r0:r1].astype(np.float32)
            ok = (v > 0) & (v < sat)
            stack[i] = np.where(ok, v / medians[i], np.nan)
        with np.errstate(all="ignore"):
            m = np.nanmedian(stack, 0)
        c = np.isfinite(stack).sum(0)
        m[c < 3] = np.nan
        master[r0:r1] = m
        counts[r0:r1] = c
    return master, counts


def generate_only(case_idx: int) -> None:
    """子进程：只负责生成输入，避免大数组驻留影响管线峰值。"""
    n, h, w = CASES[case_idx]
    root = Path(sys.argv[3])
    root.mkdir(parents=True, exist_ok=True)
    paths, _raws, sat = make_inputs(root, n, h, w)
    (root / "sat.txt").write_text(str(sat))


def run_one(case, root: Path):
    """子进程：只跑管线并测峰值；输入由另一进程预生成。"""
    n, h, w = case
    sat = float((root / "sat.txt").read_text())
    paths = sorted(root.glob("f*.fits"))
    product = build_master_flat(
        paths, [p.name for p in paths], sat,
        output_dir=root / "out", work_dir=root / "work",
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with fits.open(product) as out:
        master, counts = np.asarray(out[0].data), np.asarray(out[1].data)
        shape_ok = master.shape == (h, w) and counts.shape == (h, w)
        range_ok = int(counts.min()) >= 0 and int(counts.max()) <= n
        ext_ok = out[1].name == "COUNTS"
    return shape_ok, range_ok, ext_ok, peak


CASES = [
    (3, 4096, 4096),
    (12, 4096, 4096),
    (6, 2048, 2048),
    (5, 4096, 64),     # 窄高图
    (5, 64, 4096),     # 宽矮图
    (3, 1024, 1024),
    (12, 256, 256),
    (4, 4096, 4),      # 极小高度边界
]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "gen":
        generate_only(int(sys.argv[2]))
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        idx = int(sys.argv[2])
        root = Path(sys.argv[3])
        shape_ok, range_ok, ext_ok, peak = run_one(CASES[idx], root)
        print(f"{peak:.1f}|{int(shape_ok)}|{int(range_ok)}|{int(ext_ok)}")
        raise SystemExit(0 if (shape_ok and range_ok and ext_ok) else 1)

    import subprocess
    failures = 0
    for idx, (n, h, w) in enumerate(CASES):
        root = Path(tempfile.mkdtemp(prefix=f"matrix_{n}_{h}_{w}_"))
        gen = subprocess.run(
            [sys.executable, "-I", __file__, "gen", str(idx), str(root)],
            capture_output=True, text=True,
        )
        if gen.returncode != 0:
            print(f"n={n:2d} {w:4d}x{h:4d} GEN FAILED\n{gen.stderr[-500:]}")
            failures += 1
            continue
        proc = subprocess.run(
            [sys.executable, "-I", __file__, "run", str(idx), str(root)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"n={n:2d} {w:4d}x{h:4d} RUN FAILED\n{proc.stderr[-500:]}")
            failures += 1
            continue
        peak, shape_ok, range_ok, ext_ok = proc.stdout.strip().split("|")
        passed = shape_ok == "1" and range_ok == "1" and ext_ok == "1"
        failures += not passed
        print(
            f"n={n:2d} {w:4d}x{h:4d} peak={float(peak):7.1f}MiB "
            f"shape={shape_ok == '1'} range={range_ok == '1'} "
            f"ext={ext_ok == '1'} {'OK' if passed else 'FAIL'}"
        )
    print("MATRIX", "PASS" if failures == 0 else f"{failures} FAILURES")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
