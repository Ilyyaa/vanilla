from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.custom_rag.readers.txt_reader import TxtReader


def test_txt_reader_reads_txt_and_adds_metadata(tmp_path: Path):
    (tmp_path / "sample.txt").write_text("abcdefg", encoding="utf-8")

    reader = TxtReader(input_dir=tmp_path, chunk_size=4, chunk_overlap=1)
    chunks = reader.load_data()

    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "g"]
    assert chunks[0].metadata["file_name"] == "sample.txt"
    assert chunks[0].metadata["chunk_id"] == 0


def test_txt_reader_validates_overlap(tmp_path: Path):
    try:
        TxtReader(input_dir=tmp_path, chunk_size=100, chunk_overlap=100)
        assert False, "Expected ValueError for invalid overlap"
    except ValueError as exc:
        assert "smaller than chunk_size" in str(exc)
