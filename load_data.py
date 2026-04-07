
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from xslx_reader import XlsxReader
from app.config import get_settings

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

settings = get_settings()

embed_model = OllamaEmbedding(model_name=settings.embed_model)

pg_connection = {
    "host": settings.db_host,
    "port": settings.db_port,
    "user": settings.db_user,
    "password": settings.db_password,
    "database": settings.db_name,
}

vector_store = PGVectorStore.from_params(
    database=pg_connection["database"],
    host=pg_connection["host"],
    password=pg_connection["password"],
    port=pg_connection["port"],
    user=pg_connection["user"],
    table_name=settings.pg_table_name,
    embed_dim=settings.embed_dim,
    hybrid_search=True,
    text_search_config=settings.text_search_config,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)