from pathlib import Path
import pandas as pd
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document


class XlsxReader(BaseReader):
    def load_data(self, file, extra_info=None):
        xls = pd.ExcelFile(file)
        docs = []

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            for idx, row in df.iterrows():
                # 1. превращаем строку в текст
                row_text_parts = []
                for col, val in row.items():
                    if pd.notna(val):
                        row_text_parts.append(f"{col}: {val}")

                if not row_text_parts:
                    continue

                text = (
                    f"Название: {Path(file).stem}. " + f"Лист: {sheet_name}. "
                    + " ".join(row_text_parts)
                )

                # 2. метаданные
                metadata = {
                    "source": Path(file).stem,
                    "sheet": sheet_name,
                    "row": idx,
                }

                if extra_info:
                    metadata.update(extra_info)

                docs.append(Document(text=text, metadata=metadata))

        return docs
