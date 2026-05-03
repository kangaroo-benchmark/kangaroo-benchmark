"""Dataset access helpers for enriching row details with problem content."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import cachetools
import pyarrow.parquet as pq
from PIL import Image

from . import schemas
from .utils import build_data_url, guess_mime_from_bytes

MAX_IMAGE_DIM = 1024
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def _coerce_bytes(value) -> Optional[bytes]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=False)
        except Exception:
            return None
    return None


def _downscale_image(data: bytes) -> bytes:
    with Image.open(BytesIO(data)) as img:
        img.load()
        width, height = img.size
        max_dim = max(width, height)
        if max_dim > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max_dim
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.mode else "RGB")
        format = "PNG" if img.mode == "RGBA" else "JPEG"
        save_kwargs = {"optimize": True}
        if format == "JPEG":
            save_kwargs["quality"] = 92
        img.save(buffer, format=format, **save_kwargs)
        return buffer.getvalue()


class DatasetAccessor:
    def __init__(
        self,
        cache_size: int = 256,
        human_performance_path: Optional[Path] = None,
    ) -> None:
        self._row_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=cache_size)
        self._path_cache: Dict[str, Optional[Path]] = {}
        self._repo_root = Path(__file__).resolve().parents[1]
        self._human_performance_path = human_performance_path
        self._human_performance_data = self._load_human_data(human_performance_path)

    def _load_human_data(
        self, human_performance_path: Optional[Path]
    ) -> Dict[str, schemas.HumanPerformanceMetrics]:
        if not human_performance_path or not human_performance_path.exists():
            return {}

        try:
            table = pq.read_table(human_performance_path)
            data = table.to_pandas()
            return {
                row["question_id"]: schemas.HumanPerformanceMetrics(
                    p_correct=row.get("p_correct"),
                    sample_size=row.get("sample_size"),
                )
                for _, row in data.iterrows()
            }
        except Exception:
            # Log this properly in a real app
            return {}

    def get_row(
        self, dataset_path: Optional[str], row_id: str
    ) -> Optional[schemas.DatasetRow]:
        if not dataset_path:
            return None
        path = self._resolve_dataset_path(dataset_path)
        if not path:
            return None
        cache_key = (str(path), row_id, self._human_performance_path)
        cached = self._row_cache.get(cache_key)
        if cached:
            return cached

        record = self._read_row(path, row_id)
        if not record:
            return None
        self._row_cache[cache_key] = record
        return record

    def _resolve_dataset_path(self, dataset_path: str) -> Optional[Path]:
        if dataset_path in self._path_cache:
            return self._path_cache[dataset_path]

        raw_path = Path(dataset_path)
        candidates: List[Path] = []

        if raw_path.exists() and raw_path.is_file():
            candidates.append(raw_path)

        if not raw_path.is_absolute():
            candidates.append(self._repo_root / raw_path)
        else:
            try:
                relative = raw_path.relative_to(self._repo_root)
            except ValueError:
                relative = None
            if relative:
                candidates.append(self._repo_root / relative)

        name = raw_path.name
        if name:
            candidates.append(self._repo_root / "datasets" / name)

        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError:
                continue
            if resolved.is_file():
                self._path_cache[dataset_path] = resolved
                return resolved

        self._path_cache[dataset_path] = None
        return None

    def _read_row(self, path: Path, row_id: str) -> Optional[schemas.DatasetRow]:
        try:
            table = pq.read_table(
                path,
                filters=[("id", "==", row_id)],
                columns=[
                    "id",
                    "problem_statement",
                    "sol_A",
                    "sol_B",
                    "sol_C",
                    "sol_D",
                    "sol_E",
                    "sol_A_image_bin",
                    "sol_B_image_bin",
                    "sol_C_image_bin",
                    "sol_D_image_bin",
                    "sol_E_image_bin",
                    "question_image",
                    "associated_images_bin",
                    "language",
                    "year",
                    "group",
                    "points",
                ],
            )
        except Exception:
            return None
        if table.num_rows == 0:
            return None
        row_data = table.to_pylist()[0]
        options: List[schemas.DatasetChoice] = []
        for letter in OPTION_LETTERS:
            text = row_data.get(f"sol_{letter}")
            image_bytes = _coerce_bytes(row_data.get(f"sol_{letter}_image_bin"))
            image_url = None
            if image_bytes:
                scaled = _downscale_image(image_bytes)
                mime = guess_mime_from_bytes(scaled)
                image_url = build_data_url(scaled, mime)
            options.append(
                schemas.DatasetChoice(
                    label=letter,
                    text=str(text) if text is not None else None,
                    image=image_url,
                )
            )

        question_image = None
        q_bytes = _coerce_bytes(row_data.get("question_image"))
        if q_bytes:
            scaled = _downscale_image(q_bytes)
            mime = guess_mime_from_bytes(scaled)
            question_image = build_data_url(scaled, mime)

        associated_images: List[str] = []
        assoc_raw = row_data.get("associated_images_bin")
        if assoc_raw:
            var_iter = (
                assoc_raw if isinstance(assoc_raw, (list, tuple)) else [assoc_raw]
            )
            for candidate in var_iter:
                data = _coerce_bytes(candidate)
                if not data:
                    continue
                scaled = _downscale_image(data)
                mime = guess_mime_from_bytes(scaled)
                associated_images.append(build_data_url(scaled, mime))

        row_id_str = str(row_data.get("id"))
        human_performance = self._human_performance_data.get(row_id_str)

        return schemas.DatasetRow(
            id=row_id_str,
            problem_statement=row_data.get("problem_statement"),
            options=options,
            question_image=question_image,
            associated_images=associated_images,
            language=row_data.get("language"),
            year=str(row_data.get("year"))
            if row_data.get("year") is not None
            else None,
            group=str(row_data.get("group"))
            if row_data.get("group") is not None
            else None,
            points=float(row_data.get("points"))
            if row_data.get("points") is not None
            else None,
            human_performance=human_performance,
        )


__all__ = ["DatasetAccessor"]
