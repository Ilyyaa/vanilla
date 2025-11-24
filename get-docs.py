from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import MetadataMode
from llama_index.core import Document
import pandas as pd
from pathlib import Path

class XlsxReader(BaseReader):
    def load_data(self, file, extra_info=None):
        xls = pd.ExcelFile(file)
        docs = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            text = df.to_string()
            metadata = {
                "file_name": Path(file).stem,
                "sheet_name": sheet_name,
            }

            docs.append(Document(text=text, extra_info=metadata))
        return docs


docs = SimpleDirectoryReader(input_dir="./data", file_extractor={".xlsx": XlsxReader()}).load_data()

print(len(docs))

for doc in docs:
    # define the content/metadata template
    doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"

    # exclude page label from embedding
    if "file_path" not in doc.excluded_embed_metadata_keys:
        doc.excluded_embed_metadata_keys.append("file_path")

    if "file_name" in doc.excluded_embed_metadata_keys:
        doc.excluded_embed_metadata_keys.remove("file_name")

print(docs[0].get_content(metadata_mode=MetadataMode.EMBED))