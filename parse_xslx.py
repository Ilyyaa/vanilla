import pandas as pd
import json
from pathlib import Path
path = "data/pattern_nepreryvnost.xlsx"
excel_path = Path(path)  
output_path = excel_path.with_suffix(".jsonl")            

chunks = []
xls = pd.ExcelFile(excel_path)

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)

    df.columns = [str(c).strip() for c in df.columns]
    cols = df.columns.to_list()

    for idx, row in df.iterrows():
        param = str(row[cols[0]]).strip()
        value = str(row[cols[1]]).strip()

        text = (
            f"File: {Path(path).stem}\n\n"
            f"Sheet: {sheet_name}\n\n"
            f"Параметр: {param}\n"
            f"Значение: {value}"
        )

        chunk = {
            "id": f"{sheet_name}-{idx}",
            "metadata": {
                "file": Path(path).stem,
                "sheet": sheet_name,
                "parameter": param,
            },
            "text": text,
        }
        chunks.append(chunk)

with open(output_path, "w", encoding="utf-8") as f:
    for ch in chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")

print(f"Сохранено {len(chunks)} чанков в {output_path}")

