from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.ollama import OllamaEmbedding
from pprint import pprint

eval_set = [
  {"query": "Цель стандарта паттерн гринфилд", "gold_files": {"greenfield.xlsx"}},
  {"query": "Обязательность паттерн Критические 24×7 системы", "gold_files": {"criticality_reference.xlsx"}},
  {"query": "Способ контроля паттерн Хаускипинг и Критические 24×7 системы", "gold_files": {"hauskiping.xlsx"}},
  {"query": "когда применять паттерн интеграция", "gold_files": {"integraciya.xlsx"}},
  {"query": "когда применять паттерн консоль управления", "gold_files": {"konsol_upravleniya.xlsx"}},
  {"query": "когда применять паттерн контент", "gold_files": {"kontent.xlsx"}},
  {"query": "когда применять паттерн масштабирование", "gold_files": {"mashtabirovanie.xlsx"}},
  {"query": "когда применять паттерн отчетная база", "gold_files": {"otchetnaya_baza.xlsx"}},
  {"query": "когда применять паттерн ограничитель", "gold_files": {"ogranichitel.xlsx"}},
  {"query": "когда применять паттерн масштабирование паттерн мониторинг", "gold_files": {"monitoring.xlsx"}},
  {"query": "когда применять паттерн масштабирование паттерн прерыватель", "gold_files": {"preryvatel.xlsx"}},
  {"query": "когда применять паттерн масштабирование паттерн rate limiter", "gold_files": {"rate_limiter.xlsx"}},
  {"query": "когда применять паттерн масштабирование паттерн сегментирвоание", "gold_files": {"segmentirovanie.xlsx"}},
  {"query": "когда применять паттерн масштабирование паттерн srk", "gold_files": {"srk.xlsx"}},
  {"query": "когда применять паттерн масштабирование паттерн stand-in", "gold_files": {"stand-in.xlsx"}},
  {"query": "когда применять паттерн масштабирование паттерн standby", "gold_files": {"standby.xlsx"}},
  {"query": "когда применять паттерн масштабирование паттерн георезервирование", "gold_files": {"георезервирование.xlsx"}},
  {"query": "когда применять паттерн масштабирование паттерн непрерывность", "gold_files": {"непрерывность.xlsx"}},
  {"query": "Область видимости непрерывность", "gold_files": {"непрерывность.xlsx"}},
  {"query": "Область видимости масштабирование", "gold_files": {"mashtabirovanie.xlsx"}},
  {"query": "Область видимости контент", "gold_files": {"kontent.xlsx"}},
  {"query": "Какие есть типы БД у ИС", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "Класс критичности Интернет банка ФЛ", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "Класс критичности Интернет банка ФЛ", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx","непрерывность.xlsx"}},
  {"query": "какие паттерны применять для систем класса А", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx","непрерывность.xlsx"}},
  {"query": "какие паттерны применять для систем класса B", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx","непрерывность.xlsx"}},
  {"query": "какие паттерны применять для систем класса C", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx","непрерывность.xlsx"}},
  {"query": "Какая БД у  Бэк-офис кредитов", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "Когда применять srk", "gold_files": {"srk.xlsx"}},
  {"query": "отличие Кассового фронта от интернет банка", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "паттерн srk для Мобильный банк ФЛ", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx","srk.xlsx"}},
  {"query": "паттерн srk для Платёжный шлюз карт", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx","srk.xlsx"}},
  {"query": "паттерн srk для Система ДБО ЮЛ", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx","srk.xlsx"}},
  {"query": "паттерн srk для Бэк-офис кредитов", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx","srk.xlsx"}},
  {"query": "паттерн srk для Кассовый фронт (POS)", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx","srk.xlsx"}},
  {"query": "Общие положения непрерывность", "gold_files": {"непрерывность.xlsx"}},
  {"query": "Общие положения масштабирование", "gold_files": {"mashtabirovanie.xlsx"}},
  {"query": "Общие положения контент", "gold_files": {"kontent.xlsx"}},
  {"query": "Системы относящиеся к классу А", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "Тип нагрузки для Мобильный банк ФЛ", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "Тип нагрузки для Система ДБО ЮЛ", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "Системы относящиеся к классу B", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "Тип нагрузки для Кассовый фронт", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "чек лист для мониторинга", "gold_files": {"monitoring.xlsx"}},
  {"query": "Системы относящиеся к классу С", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "Тип нагрузки для Мобильный банк ФЛ", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "Размер БД  для Система ДБО ЮЛ", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "Критерий применимости для ограничителя", "gold_files": {"ogranichitel.xlsx"}},
  {"query": "Тип нагрузки для Отчётность для регулятора", "gold_files": {"system_patterns_reference_no_patterns_sheet.xlsx"}},
  {"query": "Чек лист для srk", "gold_files": {"srk.xlsx"}},
  {"query": "Чек лист для мониторинга", "gold_files": {"monitoring.xlsx"}},
]


def doc_recall_at_k(gold_files, retrieved_files):
    gold_files = set(gold_files)
    if not gold_files:
        return 0.0
    return len(gold_files & set(retrieved_files)) / len(gold_files)

def doc_hit_at_k(gold_files, retrieved_files):
    return 1.0 if (set(gold_files) & set(retrieved_files)) else 0.0

def eval_doc_recall(index, eval_set, ks=(1,3,5,10), query_mode="hybrid"):
    mode = VectorStoreQueryMode.HYBRID if query_mode == "hybrid" else VectorStoreQueryMode.DEFAULT

    results = {}
    for k in ks:
        retriever = index.as_retriever(
            similarity_top_k=k,
            vector_store_query_mode=mode,
        )

        recalls, hits = [], []
        for ex in eval_set:
            nodes = retriever.retrieve(ex["query"])

            retrieved_files = []
            for nws in nodes:
                fn = nws.node.metadata.get("file_name") 
                if fn:
                    retrieved_files.append(fn)
            retrieved_files = list(dict.fromkeys(retrieved_files))

            recalls.append(doc_recall_at_k(ex["gold_files"], retrieved_files))
            hits.append(doc_hit_at_k(ex["gold_files"], retrieved_files))

        results[k] = {
            "doc_recall": sum(recalls) / len(recalls),
            "doc_hit": sum(hits) / len(hits),
        }
    return results

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

embed_model = OllamaEmbedding(model_name="bge-m3")

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
    show_progress=True
)
print(eval_doc_recall(index, eval_set, ks=(1,3,5,10)))
