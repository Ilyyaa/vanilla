from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings


@dataclass(frozen=True)
class TxtChunk:
    text: str
    metadata: dict[str, str | int]


class TxtReader:
    def __init__(
        self,
        input_dir: str | Path | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()

        self.input_dir = Path(input_dir or settings.data_txt_dir)
        self.chunk_size = chunk_size if chunk_size is not None else settings.txt_chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.txt_chunk_overlap
        )

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def load_data(self) -> list[TxtChunk]:
        chunks: list[TxtChunk] = []

        for file_path in sorted(self.input_dir.glob("*.txt")):
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            for chunk_id, chunk_text in enumerate(self._chunk_text(content)):
                chunks.append(
                    TxtChunk(
                        text=chunk_text,
                        metadata={
                            "file_name": file_path.name,
                            "chunk_id": chunk_id,
                        },
                    )
                )

        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]
