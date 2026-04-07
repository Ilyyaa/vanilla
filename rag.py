from fastapi import APIRouter, status, Depends, FastAPI
from schemas import RequestIn, RequestOut
from app.config import get_settings

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from llama_index.core import PromptTemplate


custom_prompt = PromptTemplate(
    """Ты — экспертный ассистент.
Используй ТОЛЬКО следующий контекст для ответа.
Если ответа нет в контексте — скажи "не знаю".

Контекст:
{context_str}

Вопрос:
{query_str}

Ответ:
"""
)


# --- models ---
settings = get_settings()

llm = Ollama(model=settings.llm_model, request_timeout=settings.llm_request_timeout)
embed_model = OllamaEmbedding(model_name=settings.embed_model)

# --- postgres/pgvector ---
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

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)

# --- retriever (dense + sparse) ---
retriever = index.as_retriever(
    vector_store_query_mode="hybrid",
    similarity_top_k=settings.retriever_top_k,
    sparse_top_k=settings.retriever_sparse_top_k,
)

# --- cross-encoder reranker ---
reranker = SentenceTransformerRerank(
    model=settings.reranker_model,
    top_n=settings.reranker_top_n,
)

# --- query engine with reranker ---
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=llm,
    node_postprocessors=[reranker],
    text_qa_template=custom_prompt
)

# --- FastAPI ---
router = APIRouter(tags=["requests"])

@router.get("/request", status_code=status.HTTP_200_OK, response_model=RequestOut)
async def search(request: RequestIn = Depends()):
    response = query_engine.query(request.query)
    return {"answer": response.response}

app = FastAPI(title="rag", version="0.1.0")
app.include_router(router)
