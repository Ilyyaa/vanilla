from fastapi import APIRouter, status, Depends, FastAPI
from schemas import RequestIn, RequestOut

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
llm = Ollama(model="qwen2.5:1.5b", request_timeout=960.0)
embed_model = OllamaEmbedding(model_name="bge-m3") 

# --- postgres/pgvector ---
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

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)

# --- retriever (dense + sparse) ---
retriever = index.as_retriever(
    vector_store_query_mode="hybrid",
    similarity_top_k=30,  
    sparse_top_k=30,
)

# --- cross-encoder reranker ---
reranker = SentenceTransformerRerank(
    model="models/cross-encoder-russian-msmarco",  
    top_n=5,  
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
