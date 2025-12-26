---
language:
- ru
library_name: sentence-transformers
tags:
- sentence-transformers
- text-classification
- transformers
- rubert
- cross-encoder
- reranker
- msmarco
datasets:
- unicamp-dl/mmarco
base_model: DeepPavlov/rubert-base-cased
widget:
- text: как часто нужно ходить к стоматологу? [SEP] Дядя Женя работает врачем стоматологом.
  example_title: Example 1
- text: как часто нужно ходить к стоматологу? [SEP] Минимальный обязательный срок
    посещения зубного врача – раз в год, но специалисты рекомендуют делать это чаще
    – раз в полгода, а ещё лучше – раз в квартал. При таком сроке легко отследить
    любые начинающиеся проблемы и исправить их сразу же.
  example_title: Example 2
license: mit
pipeline_tag: text-ranking
---

# DiTy/cross-encoder-russian-msmarco

This is a [sentence-transformers](https://www.SBERT.net) model based on a pre-trained [DeepPavlov/rubert-base-cased](https://huggingface.co/DeepPavlov/rubert-base-cased) and finetuned with [MS-MARCO Russian passage ranking dataset](https://huggingface.co/datasets/unicamp-dl/mmarco).
The model can be used for Information Retrieval in the Russian language: Given a query, encode the query will all possible passages (e.g. retrieved with ElasticSearch). Then sort the passages in a decreasing order. See [SBERT.net Retrieve & Re-rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) for more details.

<!--- Describe your model here -->


## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder('DiTy/cross-encoder-russian-msmarco', max_length=512, device='cuda')

query = ["как часто нужно ходить к стоматологу?"]
documents = [
    "Минимальный обязательный срок посещения зубного врача – раз в год, но специалисты рекомендуют делать это чаще – раз в полгода, а ещё лучше – раз в квартал. При таком сроке легко отследить любые начинающиеся проблемы и исправить их сразу же.",
    "Основная причина заключается в истончении поверхностного слоя зуба — эмали, которая защищает зуб от механических, химических и температурных воздействий. Под эмалью расположен дентин, который более мягкий по своей структуре и пронизан множеством канальцев. При повреждении эмали происходит оголение дентинных канальцев. Раздражение с них начинает передаваться на нервные окончания в зубе и возникают болевые ощущения. Чаще всего дентин оголяется в придесневой области зубов, поскольку эмаль там наиболее тонкая и стирается быстрее.",
    "Стоматолог, также известный как стоматолог-хирург, является медицинским работником, который специализируется на стоматологии, отрасли медицины, специализирующейся на зубах, деснах и полости рта.",
    "Дядя Женя работает врачем стоматологом",
    "Плоды малины употребляют как свежими, так и замороженными или используют для приготовления варенья, желе, мармелада, соков, а также ягодного пюре. Малиновые вина, наливки, настойки, ликёры обладают высокими вкусовыми качествами.",
]

predict_result = reranker_model.predict([[query[0], documents[0]]])
print(predict_result)
# `array([0.88126713], dtype=float32)`

rank_result = reranker_model.rank(query[0], documents)
print(rank_result)
# `[{'corpus_id': 0, 'score': 0.88126713},
#  {'corpus_id': 2, 'score': 0.001042091},
#  {'corpus_id': 3, 'score': 0.0010417715},
#  {'corpus_id': 1, 'score': 0.0010344835},
#  {'corpus_id': 4, 'score': 0.0010244923}]`
```


## Usage (HuggingFace Transformers)
Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you need to get the logits from the model.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('DiTy/cross-encoder-russian-msmarco')
tokenizer = AutoTokenizer.from_pretrained('DiTy/cross-encoder-russian-msmarco')

features = tokenizer(["как часто нужно ходить к стоматологу?", "как часто нужно ходить к стоматологу?"], ["Минимальный обязательный срок посещения зубного врача – раз в год, но специалисты рекомендуют делать это чаще – раз в полгода, а ещё лучше – раз в квартал. При таком сроке легко отследить любые начинающиеся проблемы и исправить их сразу же.", "Дядя Женя работает врачем стоматологом"], padding=True, truncation=True, return_tensors='pt')
 
model.eval()
with torch.no_grad():
    scores = model(**features).logits
    print(scores)
# `tensor([[ 1.6871],
#        [-6.8700]])`
```