
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.core.readers.base import BaseReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from xslx_reader import XlsxReader

docs = SimpleDirectoryReader(
    input_dir="./data",
    file_extractor={".xlsx": XlsxReader()},
).load_data()

for doc in docs:
    doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"

    if "file_path" not in doc.excluded_embed_metadata_keys:
        doc.excluded_embed_metadata_keys.append("file_path")

    if "file_name" in doc.excluded_embed_metadata_keys:
        doc.excluded_embed_metadata_keys.remove("file_name")

embed_model = OllamaEmbedding(model_name="bge-m3")

pg_connection = {
    "host": "localhost",
    "port": 5433,
    "user": "rag_user",
    "password": "password",
    "database": "rag_db",
}

vector_store = PGVectorStore.from_params(
    database=pg_connection["database"],
    host=pg_connection["host"],
    password=pg_connection["password"],
    port=pg_connection["port"],
    user=pg_connection["user"],
    table_name="llamaindex_vectors",
    embed_dim=1024,
    hybrid_search=True,           
    text_search_config="russian",
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)