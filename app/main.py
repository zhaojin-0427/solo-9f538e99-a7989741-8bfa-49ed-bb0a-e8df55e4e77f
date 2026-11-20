"""主平场制备 FastAPI 接口。

* ``GET  /docs``    Swagger 交互文档，可直接上传 FITS。
* ``GET  /health``  存活探针。
* ``POST /prepare`` 上传 3~12 张平场 FITS 与饱和阈值，返回成品 FITS。
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .pipeline import (
    MAX_FRAMES,
    MIN_FRAMES,
    PreparationError,
    build_master_flat,
)

logger = logging.getLogger("master_flat")

app = FastAPI(
    title="天文测光主平场制备接口",
    version="1.0.0",
    description=(
        "上传 3~12 张已扣偏置的无压缩二维 16 位整数平场 FITS，"
        "按 BLANK / BSCALE / BZERO 处理后进行每帧中位数归一与像素级中位数聚合，"
        "输出含校正系数主图与参与帧数扩展图的成品 FITS。"
    ),
)

# 单容器内存预算 256 MiB，串行处理避免并发制备叠加内存。
_preparation_lock = threading.Lock()


def _data_dir(env_name: str, default: str) -> Path:
    path = Path(__file__).resolve().parent.parent / os.environ.get(env_name, default)
    path.mkdir(parents=True, exist_ok=True)
    return path


WORK_DIR = _data_dir("FLAT_WORK_DIR", "work")
OUTPUT_DIR = _data_dir("FLAT_OUTPUT_DIR", "products")
UPLOAD_CHUNK = 4 * 2 ** 20


@app.exception_handler(PreparationError)
async def preparation_error_handler(
    request: Request, exc: PreparationError
) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/prepare", response_class=FileResponse)
async def prepare_master_flat(
    files: list[UploadFile] = File(
        ...,
        description=f"平场 FITS 文件，数量 {MIN_FRAMES}~{MAX_FRAMES} 张",
    ),
    saturation: float = Form(
        ..., description="正且有限的饱和阈值（与 BSCALE/BZERO 换算后同单位）"
    ),
) -> FileResponse:
    if not (MIN_FRAMES <= len(files) <= MAX_FRAMES):
        raise PreparationError(
            f"平场帧数须在 {MIN_FRAMES}~{MAX_FRAMES} 张之间，收到 {len(files)} 张"
        )

    batch_dir = Path(tempfile.mkdtemp(prefix="batch_", dir=WORK_DIR))
    frame_paths: list[Path] = []
    source_names: list[str] = []
    try:
        for index, upload in enumerate(files, start=1):
            if not upload.filename:
                raise PreparationError(f"第 {index} 个上传项缺少文件名")
            # 每个上传文件放在独立的新空目录中，隔离不可信数据。
            frame_dir = batch_dir / f"frame_{index:02d}"
            frame_dir.mkdir()
            safe_name = Path(upload.filename).name or f"frame_{index:02d}.fits"
            target = frame_dir / safe_name
            try:
                with target.open("wb") as out:
                    while True:
                        chunk = await upload.read(UPLOAD_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
            except OSError as exc:
                raise PreparationError(f"上传文件保存失败：{upload.filename}：{exc}") from exc
            finally:
                await upload.close()
            if target.stat().st_size == 0:
                raise PreparationError(f"上传文件为空：{upload.filename}")
            frame_paths.append(target)
            source_names.append(upload.filename)

        with _preparation_lock:
            product = build_master_flat(
                frame_paths=frame_paths,
                source_names=source_names,
                saturation=saturation,
                output_dir=OUTPUT_DIR,
                work_dir=WORK_DIR,
            )

        logger.info("主平场制备成功：%s <- %d 帧", product.name, len(files))
        return FileResponse(
            product,
            media_type="application/fits",
            filename=product.name,
            headers={"X-Frame-Count": str(len(files))},
        )
    finally:
        shutil.rmtree(batch_dir, ignore_errors=True)
