"""主平场（master flat）制备管线。

设计要点
========

* 输入为 3~12 张已扣偏置、无压缩、二维 16 位整数单图 FITS。
* 用 ``do_not_scale_image_data=True`` 读入原始整数，自行按
  ``BSCALE`` / ``BZERO`` 换算，显式按 ``BLANK`` 排除缺测像元。
* 有效像元：换算后 ``0 < value < saturation``（且不等于 BLANK）。
* 每帧有效值先除以该帧有效像元中位数；再对同一位置的有效归一值取中位数；
  参与帧数少于 3 的位置输出 NaN。
* 16 位整数只有 65536 个取值，第一趟用直方图精确求每帧中位数，
  避免一次性存放整帧有效值。
* 第二趟按列条带聚合，条带宽度依据图像尺寸自适应，把常驻内存控制在
  256 MiB 容器预算内；每个文件仅重读有限次数。
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

MIN_FRAMES = 3
MAX_FRAMES = 12
MAX_DIM = 4096
MIN_FRAMES_PER_PIXEL = 3
# 条带聚合缓冲上限（float32）。取值需为排序/花式索引等瞬时分配、
# 常驻结果图与解释器基线留出余量，使总峰值稳在 256 MiB 容器预算内。
STRIP_BUDGET_BYTES = 8 * 2 ** 20
# 单次行块的行数，限制换算时的临时数组大小。
ROW_CHUNK = 256
RAW_BIN_COUNT = 65536  # int16 全部取值数
RAW_OFFSET = 32768
FITS_BLOCK = 2880  # FITS 头部与数据块的最小对齐单位


class PreparationError(Exception):
    """整批制备失败；status_code 决定 HTTP 响应码。"""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class FrameInfo:
    """单帧的元数据与第一趟统计结果。"""

    path: Path
    source_name: str
    shape: tuple[int, int]
    image_index: int
    bscale: float
    bzero: float
    blank: int | None
    median: float


def _open_raw(path: Path) -> fits.HDUList:
    """打开 FITS 且禁止 astropy 自动换算，保留原始整数数据。"""
    try:
        return fits.open(path, do_not_scale_image_data=True, memmap=False)
    except (OSError, ValueError, fits.VerifyError) as exc:
        raise PreparationError(f"FITS 文件截断或无法解析：{path.name}：{exc}") from exc


def _read_raw(path: Path, image_index: int) -> np.ndarray:
    """打开并实体化原始整数主图；任何读取异常都按整批可预期失败处理。"""
    try:
        with _open_raw(path) as hdul:
            return np.asarray(hdul[image_index].data)
    except (OSError, ValueError, EOFError, fits.VerifyError) as exc:
        raise PreparationError(f"FITS 文件截断或数据损坏：{path.name}：{exc}") from exc


def _validate_hdus(hdul: fits.HDUList, name: str) -> tuple[int, fits.ImageHDU]:
    """校验每张图恰好包含一幅无压缩二维 16 位整数主图，返回其 HDU 序号。"""
    image_hdus: list[tuple[int, fits.ImageHDU]] = []
    for index, hdu in enumerate(hdul):
        if isinstance(hdu, fits.CompImageHDU):
            raise PreparationError(f"帧 {name} 含压缩图像 HDU，仅接受无压缩 FITS")
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            raise PreparationError(f"帧 {name} 含表 HDU，每张 FITS 只能有一幅主图")
        try:
            hdu_data = hdu.data
        except (OSError, ValueError, EOFError) as exc:
            raise PreparationError(f"FITS 文件截断或数据损坏：{name}：{exc}") from exc
        if hdu.is_image and hdu_data is not None:
            image_hdus.append((index, hdu))

    if len(image_hdus) != 1:
        raise PreparationError(
            f"帧 {name} 含 {len(image_hdus)} 幅图像，要求每张 FITS 恰好一幅主图"
        )

    index, hdu = image_hdus[0]
    bitpix = hdu.header.get("BITPIX")
    if bitpix != 16:
        raise PreparationError(f"帧 {name} 的 BITPIX={bitpix}，要求为 16 位整数（BITPIX=16）")

    try:
        data = np.asarray(hdu_data)
    except (OSError, ValueError) as exc:
        raise PreparationError(f"FITS 文件截断或数据损坏：{name}：{exc}") from exc
    if data.ndim != 2:
        raise PreparationError(f"帧 {name} 主图为 {data.ndim} 维，要求二维图像")
    if data.dtype.kind != "i" or data.dtype.itemsize != 2:
        raise PreparationError(f"帧 {name} 主图数据类型 {data.dtype} 不是 16 位整数")

    height, width = data.shape
    if height == 0 or width == 0 or height > MAX_DIM or width > MAX_DIM:
        raise PreparationError(
            f"帧 {name} 尺寸 {width}x{height} 非法，宽高均须在 1~{MAX_DIM} 之间"
        )
    return index, hdu


def _camera_filter_keys(header: fits.Header) -> dict[str, str]:
    """提取相机/滤镜标识；缺失关键字不参与比对（不强制要求这些头存在）。"""
    result: dict[str, str] = {}
    for key in ("INSTRUME", "FILTER"):
        if key in header and header[key] is not None:
            value = str(header[key]).strip()
            if value:
                result[key] = value
    return result


def _read_scaling(header: fits.Header, name: str) -> tuple[float, float, int | None]:
    try:
        bscale = float(header.get("BSCALE", 1.0))
        bzero = float(header.get("BZERO", 0.0))
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"帧 {name} 的 BSCALE/BZERO 关键字非法：{exc}") from exc
    if not (math.isfinite(bscale) and math.isfinite(bzero)) or bscale == 0.0:
        raise PreparationError(f"帧 {name} 的 BSCALE/BZERO 非有限值或 BSCALE 为 0")

    blank: int | None = None
    if "BLANK" in header:
        try:
            blank = int(header["BLANK"])
        except (TypeError, ValueError) as exc:
            raise PreparationError(f"帧 {name} 的 BLANK 关键字不是整数：{exc}") from exc
        if not (-32768 <= blank <= 32767):
            raise PreparationError(f"帧 {name} 的 BLANK={blank} 超出 16 位整数范围")
    return bscale, bzero, blank


def _frame_median(
    path: Path,
    image_index: int,
    bscale: float,
    bzero: float,
    blank: int | None,
    saturation: float,
) -> tuple[int, np.ndarray]:
    """第一趟：基于原始值直方图求有效像元中位数。

    返回 ``(有效像元总数, 65536 桶直方图)``，由调用方换算精确中位数，
    避免一次性存放整帧有效值。
    """
    hist = np.zeros(RAW_BIN_COUNT, dtype=np.int64)
    raw = _read_raw(path, image_index)

    for r0 in range(0, raw.shape[0], ROW_CHUNK):
        block = raw[r0 : r0 + ROW_CHUNK].astype(np.int32, copy=False)
        scaled = block.astype(np.float64) * bscale + bzero
        valid = (scaled > 0.0) & (scaled < saturation)
        if blank is not None:
            valid &= block != blank
        if not valid.any():
            continue
        indices = block + RAW_OFFSET
        hist += np.bincount(indices[valid], minlength=RAW_BIN_COUNT)

    total = int(hist.sum())
    if total == 0:
        raise PreparationError(f"帧 {path.name} 不存在任何有效像元，整批制备失败")
    return total, hist


def _median_scaled_from_hist(
    hist: np.ndarray, total: int, bscale: float, bzero: float
) -> float:
    """由原始整数直方图精确换算有效像元（缩放后）的中位数。"""
    k1 = (total - 1) // 2
    k2 = total // 2
    ordered = hist if bscale > 0 else hist[::-1]
    cumulative = np.cumsum(ordered)
    p1 = int(np.searchsorted(cumulative, k1 + 1, side="left"))
    p2 = int(np.searchsorted(cumulative, k2 + 1, side="left"))
    if bscale > 0:
        raw1, raw2 = p1 - RAW_OFFSET, p2 - RAW_OFFSET
    else:
        raw1, raw2 = RAW_OFFSET - 1 - p1, RAW_OFFSET - 1 - p2
    return ((raw1 * bscale + bzero) + (raw2 * bscale + bzero)) / 2.0


def _strip_width(n_frames: int, height: int) -> int:
    """根据内存预算计算列条带宽度（float32 聚合同步缓冲）。"""
    per_column = n_frames * height * np.dtype(np.float32).itemsize
    width = STRIP_BUDGET_BYTES // max(per_column, 1)
    return max(1, min(width, MAX_DIM))


def _aggregate_buffer(
    buffer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """对帧轴求有效归一值中位数与有效帧数。

    不用 ``np.nanmedian``：它会对整个条带再复制一份并做类型提升，
    在 256 MiB 预算下峰值超标。这里原地排序（NaN 按浮点规则排到末尾），
    再按每像素的有效计数用索引直接取中位数。
    """
    n_frames = buffer.shape[0]
    counts = np.isfinite(buffer).sum(axis=0, dtype=np.uint8)

    buffer.sort(axis=0)  # 原地，避免整份条带的复制与 float64 提升
    median_strip = np.full(buffer.shape[1:], np.nan, dtype=np.float32)
    valid_pixels = counts > 0
    rows, cols = np.nonzero(valid_pixels)
    k1 = ((counts[valid_pixels] - 1) // 2).astype(np.int64)
    k2 = (counts[valid_pixels] // 2).astype(np.int64)
    # 偶数计数取中间两个均值；奇数计数两索引相同，等价直接取值。
    median_strip[valid_pixels] = (
        buffer[k1, rows, cols].astype(np.float64)
        + buffer[k2, rows, cols].astype(np.float64)
    ) / 2.0
    median_strip[counts < MIN_FRAMES_PER_PIXEL] = np.nan
    return median_strip, counts


def _fill_strip(
    frames: list[FrameInfo],
    col0: int,
    col1: int,
    saturation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """聚合一个列条带：返回归一值中位数条带与参与帧数条带。"""
    height = frames[0].shape[0]
    strip_w = col1 - col0
    buffer = np.full((len(frames), height, strip_w), np.nan, dtype=np.float32)

    for f_index, frame in enumerate(frames):
        raw = _read_raw(frame.path, frame.image_index)
        for r0 in range(0, height, ROW_CHUNK):
            r1 = min(r0 + ROW_CHUNK, height)
            block = raw[r0:r1, col0:col1].astype(np.int32, copy=False)
            scaled = block.astype(np.float64) * frame.bscale + frame.bzero
            valid = (scaled > 0.0) & (scaled < saturation)
            if frame.blank is not None:
                valid &= block != frame.blank
            normalized = (scaled / frame.median).astype(np.float32)
            target = buffer[f_index, r0:r1, :]
            np.copyto(target, normalized, where=valid)

    return _aggregate_buffer(buffer)


def _sanitize_ascii(text: str, limit: int = 60) -> str:
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in text).strip()
    return cleaned[:limit] or "unknown"


def build_master_flat(
    frame_paths: list[Path],
    source_names: list[str],
    saturation: float,
    output_dir: Path,
    work_dir: Path,
) -> Path:
    """制备主平场，成功后返回成品 FITS 的最终路径，失败则整体报错。

    输出文件先写到 ``work_dir`` 内的临时文件，校验通过后原子移动到
    ``output_dir``；任何一步失败都不会留下半成品。
    """
    if not (math.isfinite(saturation) and saturation > 0.0):
        raise PreparationError("饱和阈值必须为正且有限的数值")
    if not (MIN_FRAMES <= len(frame_paths) <= MAX_FRAMES):
        raise PreparationError(
            f"平场帧数须在 {MIN_FRAMES}~{MAX_FRAMES} 张之间，收到 {len(frame_paths)} 张"
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PreparationError(f"工作目录不可用，整批制备失败：{exc}", status_code=500) from exc

    frames: list[FrameInfo] = []
    common_shape: tuple[int, int] | None = None
    common_camera_filter: dict[str, str] | None = None

    # 第一趟：结构校验 + 每帧有效值直方图与中位数。
    for path, source_name in zip(frame_paths, source_names):
        with _open_raw(path) as hdul:
            image_index, hdu = _validate_hdus(hdul, path.name)
            shape = (int(hdu.data.shape[0]), int(hdu.data.shape[1]))
            bscale, bzero, blank = _read_scaling(hdu.header, path.name)
            camera_filter = _camera_filter_keys(hdu.header)

        if common_shape is None:
            common_shape = shape
        elif shape != common_shape:
            raise PreparationError(
                f"帧 {path.name} 尺寸 {shape[1]}x{shape[0]} 与首帧 "
                f"{common_shape[1]}x{common_shape[0]} 不一致，同批尺寸须一致"
            )

        if common_camera_filter is None:
            common_camera_filter = camera_filter
        elif camera_filter != common_camera_filter:
            raise PreparationError(
                f"帧 {path.name} 的相机/滤镜标识 {camera_filter} 与首帧 "
                f"{common_camera_filter} 不一致，同批相机与滤镜须一致"
            )

        total, hist = _frame_median(
            path, image_index, bscale, bzero, blank, saturation
        )
        median = _median_scaled_from_hist(hist, total, bscale, bzero)
        if not math.isfinite(median) or median <= 0.0:
            raise PreparationError(f"帧 {path.name} 的有效中位数非法（{median}）")
        frames.append(
            FrameInfo(path, source_name, shape, image_index, bscale, bzero, blank, median)
        )

    assert common_shape is not None
    height, width = common_shape
    n_frames = len(frames)

    master = np.full((height, width), np.nan, dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint8)

    # 第二趟：自适应列条带聚合，限制常驻内存。
    strip_w = _strip_width(n_frames, height)
    for col0 in range(0, width, strip_w):
        col1 = min(col0 + strip_w, width)
        median_strip, count_strip = _fill_strip(
            frames, col0, col1, saturation
        )
        master[:, col0:col1] = median_strip
        counts[:, col0:col1] = count_strip

    return _write_output(frames, saturation, master, counts, output_dir, work_dir)


def _write_fits_low_memory(
    master: np.ndarray,
    counts: np.ndarray,
    header0: fits.Header,
    header1: fits.Header,
    tmp_path: Path,
) -> None:
    """流式写出双 HDU FITS，避免 ``HDUList.writeto`` 的整图内部缓冲。

    主图 float32（BITPIX=-32），扩展图 uint8（BITPIX=8，帧数最多 12）。
    头部仍由 astropy 生成以保证关键字合规；数据块用 ``ndarray.tofile``
    直接按大端字节追加，并补齐到 2880 字节块。
    """

    def header_block(header: fits.Header) -> bytes:
        # 直接序列化调用方基于真实数组建好的头部；切勿再用
        # PrimaryHDU(header=...) 重建，那会把 BITPIX/NAXIS 重置。
        # padding=True 保证输出为整 2880 字节块且 END 卡对齐卡边界。
        return header.tostring(sep="", padding=True).encode("ascii")

    blocks = (header_block(header0), header_block(header1))

    with tmp_path.open("wb") as handle:
        handle.write(blocks[0])
        np.ascontiguousarray(master, dtype=">f4").tofile(handle)
        pad = (FITS_BLOCK - master.nbytes % FITS_BLOCK) % FITS_BLOCK
        if pad:
            handle.write(b"\0" * pad)
        handle.write(blocks[1])
        np.ascontiguousarray(counts, dtype=np.uint8).tofile(handle)
        pad = (FITS_BLOCK - counts.nbytes % FITS_BLOCK) % FITS_BLOCK
        if pad:
            handle.write(b"\0" * pad)
        handle.flush()
        os.fsync(handle.fileno())


def _write_output(
    frames: list[FrameInfo],
    saturation: float,
    master: np.ndarray,
    counts: np.ndarray,
    output_dir: Path,
    work_dir: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    token = os.urandom(3).hex()
    final_name = f"master_flat_{timestamp}_{token}.fits"
    tmp_path = work_dir / f".{final_name}.tmp"
    final_path = output_dir / final_name

    primary_hdu = fits.PrimaryHDU(data=master)
    header = primary_hdu.header
    header["IMAGTYPE"] = ("MASTERFLAT", "Master flat field correction coefficients")
    header["NFRAMES"] = (len(frames), "Number of input flat frames")
    header["SATURATE"] = (saturation, "Saturation threshold in data units")
    header["MINCNT"] = (MIN_FRAMES_PER_PIXEL, "Minimum frames for a valid pixel")
    header["CREATED"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Creation time (UTC)",
    )
    for idx, frame in enumerate(frames):
        header[f"MED{idx + 1:02d}"] = (
            frame.median,
            f"Valid-pixel median of frame {idx + 1}",
        )
    header.add_history(
        "Pixels scaled by BSCALE/BZERO; BLANK, non-positive and saturated "
        "pixels excluded; per-frame median normalization followed by pixelwise "
        "median; pixels seen by fewer than 3 frames are NaN."
    )
    for idx, frame in enumerate(frames):
        header.add_history(f"Input {idx + 1:02d}: {_sanitize_ascii(frame.source_name)}")

    counts_hdu = fits.ImageHDU(data=counts, name="COUNTS")
    # 取头部副本后立即释放 HDU，避免 HDU 额外持有数据引用。
    header0, header1 = header.copy(), counts_hdu.header.copy()
    del primary_hdu, counts_hdu, header

    try:
        _write_fits_low_memory(master, counts, header0, header1, tmp_path)
        # 头部回读校验结构与尺寸（不实体化数据，避免额外内存），
        # 再用 mmap 抽样确认数据块可解析。
        with fits.open(tmp_path, memmap=False) as verify:
            if len(verify) != 2:
                raise PreparationError("输出 FITS 结构校验失败", status_code=500)
            if verify[0].shape != master.shape or verify[1].shape != counts.shape:
                raise PreparationError("输出 FITS 尺寸校验失败", status_code=500)
            if verify[1].name != "COUNTS":
                raise PreparationError("输出 FITS 扩展名校验失败", status_code=500)
        with fits.open(tmp_path, memmap=True) as verify:
            sample0 = np.asarray(verify[0].data)
            sample1 = np.asarray(verify[1].data)
            if sample0.shape != master.shape or sample1.shape != counts.shape:
                raise PreparationError("输出 FITS 数据块校验失败", status_code=500)
            # 帧数图取值必须落在 0~NFRAMES；确认数据块整体可读。
            if int(sample1.min()) < 0 or int(sample1.max()) > len(frames):
                raise PreparationError("输出 FITS 数据块校验失败", status_code=500)
        os.replace(tmp_path, final_path)
    except PreparationError:
        tmp_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise PreparationError(f"主平场写出失败：{exc}", status_code=500) from exc

    return final_path
