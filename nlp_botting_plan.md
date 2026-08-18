# План изучения NLP по книге «Обработка естественного языка в действии», 2-е издание

## Цель

Пройти книгу не как набор теории, а как практический курс NLP-инженера:

- понять, как текст превращается в признаки и векторы;
- научиться строить baseline-модели;
- освоить semantic search, embeddings, CNN, RNN и Transformers;
- сравнивать классические NLP-подходы с LLM;
- получить один основной эволюционирующий проект и несколько дополнительных мини-проектов;
- довести итоговый проект до API и Docker.

---

# Основной проект: классификатор обращений

Основной датасет:

**BANKING77**  
https://huggingface.co/datasets/PolyAI/banking77

Задача: по тексту обращения клиента определить один из 77 интентов.

Примеры классов:

- `card_not_working`
- `cash_withdrawal_not_recognised`
- `lost_or_stolen_card`
- `transaction_charged_twice`
- `cash_withdrawal`
- `declined_card_payment`

Загрузка:

```python
from datasets import load_dataset

dataset = load_dataset("PolyAI/banking77")
```

Для начала можно взять 10–15 классов, а затем перейти ко всем 77.

---

# Глава 2. Токены и предобработка

## Что понять

- token;
- vocabulary;
- word tokenizer;
- character tokenizer;
- n-grams;
- stemming;
- lemmatization;
- stop words;
- OOV;
- влияние preprocessing на задачу.

Главный вопрос главы:

> Как текст превращается в последовательность объектов, которую может использовать модель?

## Практика

На BANKING77:

1. Посмотреть структуру датасета.
2. Посчитать количество документов.
3. Посмотреть распределение классов.
4. Посчитать длину текстов.
5. Получить:
   - среднюю длину;
   - median;
   - p95;
   - максимальную длину.
6. Сравнить токенизацию:
   - `str.split()`;
   - NLTK;
   - spaCy.
7. Построить vocabulary.
8. Найти самые частые unigram.
9. Найти самые частые bigram.
10. Проверить влияние:
    - lowercase;
    - punctuation removal;
    - stopwords;
    - stemming;
    - lemmatization.
11. Сравнить размер vocabulary после каждого шага.

## Результат

Пример структуры:

```text
src/
    preprocessing.py

notebooks/
    01_dataset_analysis.ipynb
```

После главы нужно понимать путь:

```text
raw string
    ↓
normalization
    ↓
tokenization
    ↓
tokens
    ↓
vocabulary
    ↓
token IDs / features
```

---

# Глава 3. Bag of Words и TF-IDF

## Что понять

- Bag of Words;
- term frequency;
- inverse document frequency;
- TF-IDF;
- sparse vectors;
- cosine similarity;
- n-grams как признаки.

## Практика: первый baseline

Построить:

```text
text
 ↓
TfidfVectorizer
 ↓
LogisticRegression
 ↓
intent
```

Пример:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
        ),
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=1000,
        ),
    ),
])
```

## Метрики

Посчитать:

- accuracy;
- macro F1;
- confusion matrix.

## Error Analysis

Обязательно посмотреть минимум 30–50 ошибок.

Для каждой ошибки сохранить:

```text
Text:
...

Target:
...

Prediction:
...

Почему модель могла ошибиться:
...
```

Главный навык:

> Не только получить метрику, но и понять причины ошибок модели.

---

# Глава 4. Семантика через частотные методы

## Что понять

- dimensionality reduction;
- SVD;
- latent semantic analysis;
- latent semantic vectors;
- отличие TF-IDF-вектора от плотного семантического представления.

## Практика

Сделать:

```text
TF-IDF
  ↓
TruncatedSVD
  ↓
dense document vector
```

Например:

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=100)
```

## Мини-задача: поиск похожих обращений

Для запроса:

```text
my card hasn't arrived
```

найти 5–10 наиболее похожих обращений через cosine similarity.

Получится первый простой retrieval pipeline:

```text
query
 ↓
vectorization
 ↓
similarity
 ↓
nearest documents
```

---

# Глава 5. Нейронные сети

## Что понять

- input layer;
- hidden layers;
- activation;
- logits;
- loss;
- backpropagation;
- optimizer;
- batch;
- epoch;
- overfitting.

## Практика

Не делать новый проект.

Использовать тот же BANKING77:

```text
TF-IDF
 ↓
PyTorch MLP
 ↓
77 classes
```

Пример архитектуры:

```text
20000
 ↓
512
 ↓
128
 ↓
77
```

## Главный вывод

Сравнить MLP с Logistic Regression.

Нужно увидеть на практике:

> Более сложная нейросеть не обязана быть лучше хорошего линейного baseline.

---

# Глава 6. Word Embeddings

## Что понять

Переход:

```text
слово → индекс
```

к:

```text
слово → dense vector
```

Изучить:

- Word2Vec;
- embedding space;
- cosine similarity;
- semantic neighbourhood;
- pooling нескольких word embeddings.

## Практика 1: Word2Vec

Посмотреть ближайшие слова:

```python
model.wv.most_similar("payment")
model.wv.most_similar("card")
model.wv.most_similar("money")
```

## Практика 2: классификация через embeddings

Сделать:

```text
sentence
 ↓
word embeddings
 ↓
mean pooling
 ↓
sentence vector
 ↓
classifier
```

Сравнить с TF-IDF.

## Таблица экспериментов

Начать вести единую таблицу:

| Model | Macro F1 | Accuracy | Latency | Notes |
|---|---:|---:|---:|---|
| TF-IDF + Logistic Regression | | | | |
| TF-IDF + MLP | | | | |
| Word2Vec + classifier | | | | |

---

# Дополнительная практика после главы 6: Semantic Similarity

Датасет:

**STS-B**  
https://huggingface.co/datasets/sentence-transformers/stsb

Задача:

```text
sentence A ─→ embedding ─┐
                         ├→ cosine similarity
sentence B ─→ embedding ─┘
```

Сравнивать similarity модели с эталонным human similarity score.

## Что понять

- sentence embeddings;
- semantic similarity;
- cosine similarity;
- разницу между word-level и sentence-level embeddings.

---

# Глава 7. CNN для текста

## Что понять

- Conv1D для текста;
- локальные паттерны;
- kernel size;
- pooling;
- почему CNN хорошо ловит короткие устойчивые выражения.

Примеры:

```text
lost card
cash withdrawal
exchange rate
cash deposit
```

## Практика

Реализовать `TextCNN` на PyTorch:

```text
Embedding
   ↓
Conv1D kernel=2
Conv1D kernel=3
Conv1D kernel=4
   ↓
GlobalMaxPool
   ↓
Linear
   ↓
77 classes
```

Добавить результат в общую таблицу.

---

# Глава 8. RNN / LSTM

## Что понять

- последовательная обработка;
- hidden state;
- RNN;
- LSTM;
- bidirectional LSTM;
- padding;
- packed sequences;
- влияние порядка слов.

## Практика

```text
tokens
 ↓
Embedding
 ↓
BiLSTM
 ↓
Linear
 ↓
77 classes
```

## Сравнение

К этому моменту таблица должна выглядеть примерно так:

| Approach | Macro F1 | Params | Latency |
|---|---:|---:|---:|
| TF-IDF + LR | | | |
| TF-IDF + MLP | | | |
| Word2Vec + classifier | | | |
| TextCNN | | | |
| BiLSTM | | | |

---

# Глава 9. Transformers

## Что понять

- attention;
- self-attention;
- positional encoding;
- tokenizer;
- subword tokenization;
- contextual embeddings;
- CLS token;
- encoder;
- fine-tuning.

## Практика

Использовать BANKING77:

```text
text
 ↓
Transformer tokenizer
 ↓
BERT / DistilBERT
 ↓
CLS representation
 ↓
classification head
 ↓
77 classes
```

Подходящие модели для начала:

- DistilBERT;
- BERT-base.

## Главная задача

Сравнить Transformer с тем, что уже реализовано:

```text
TF-IDF
Word2Vec
CNN
BiLSTM
Transformer
```

Теперь Transformer будет восприниматься как следующий этап развития методов, а не как магический black box.

---

# Глава 10. LLM

## Что понять

- zero-shot classification;
- few-shot classification;
- prompt design;
- structured output;
- inference cost;
- latency;
- stability;
- когда LLM не нужна;
- когда fine-tuning оправдан.

## Практика

Взять 200–500 примеров из test BANKING77.

Попросить LLM выбрать ровно один intent.

Пример задачи:

```text
Classify this customer request into exactly one intent.

Request:
"I was charged twice for my card payment."

Classes:
...
```

Сравнить:

```text
TF-IDF + Logistic Regression
BERT fine-tuned
LLM zero-shot
LLM few-shot
```

## Сравнивать не только качество

Записывать:

- Macro F1;
- latency;
- стоимость;
- стабильность;
- количество размеченных данных;
- сложность поддержки.

Главный инженерный вопрос:

> Какой метод рациональнее для конкретной задачи?

---

# Дополнительный проект: Question Answering

Датасет:

**SQuAD**  
https://huggingface.co/datasets/rajpurkar/squad

## Практика

```text
document
 +
question
 ↓
QA model
 ↓
answer span
```

## Что понять

Разницу между:

```text
classification
retrieval
extractive QA
generation
```

---

# Глава 11. Information Extraction / NER

Датасет:

**MultiNERD**  
https://huggingface.co/datasets/Babelscape/multinerd

В датасете есть русский язык.

## Что понять

- Named Entity Recognition;
- token classification;
- BIO tagging;
- span extraction;
- entity types;
- postprocessing.

Пример:

```text
Иван   B-PER
Петров I-PER
работает O
в       O
Яндекс  B-ORG
```

## Практика 1

Попробовать готовый NER pipeline, например spaCy или Transformer.

## Практика 2

Дообучить Transformer под token classification.

## Практика 3

Сделать нормальный API-ответ:

```json
{
  "entities": [
    {
      "text": "Иван Петров",
      "type": "PERSON"
    },
    {
      "text": "Яндекс",
      "type": "ORGANIZATION"
    }
  ]
}
```

---

# Глава 12. Диалоговые системы

Дополнительный датасет:

**DailyDialog**  
https://huggingface.co/datasets/roskoN/dailydialog

## Не нужно сразу делать ChatGPT

Сначала построить классический pipeline:

```text
User message
 ↓
intent classifier
 ↓
intent
 ↓
dialog manager
 ↓
response
```

Например:

```text
User:
"My card was stolen"

 ↓

intent:
lost_or_stolen_card

 ↓

next_action:
block_card

 ↓

response
```

## Добавить состояние

```text
intent
slots
history
next_action
```

После этого сравнить rule-based/dialog-manager подход с LLM-based chatbot.

---

# Финальный этап: Production NLP Service

После основной части книги перестать работать только в notebook.

Сделать сервис:

```text
POST /predict
```

Пример запроса:

```json
{
  "text": "My cash withdrawal was declined"
}
```

Пример ответа:

```json
{
  "intent": "declined_cash_withdrawal",
  "confidence": 0.93
}
```

## Структура

```text
nlp-banking77/
│
├── src/
│   ├── preprocessing.py
│   ├── inference.py
│   ├── models/
│   └── metrics.py
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_tfidf_baseline.ipynb
│   ├── 03_svd.ipynb
│   ├── 04_embeddings.ipynb
│   ├── 05_cnn.ipynb
│   ├── 06_lstm.ipynb
│   └── 07_transformer.ipynb
│
├── tests/
│
├── app/
│   └── main.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Стек

- Python;
- pandas;
- NumPy;
- scikit-learn;
- PyTorch;
- Hugging Face Transformers;
- Hugging Face Datasets;
- spaCy;
- FastAPI;
- Docker.

---

# Основные датасеты

| Направление | Датасет | Ссылка |
|---|---|---|
| Classification | BANKING77 | https://huggingface.co/datasets/PolyAI/banking77 |
| Semantic Similarity | STS-B | https://huggingface.co/datasets/sentence-transformers/stsb |
| Question Answering | SQuAD | https://huggingface.co/datasets/rajpurkar/squad |
| NER | MultiNERD | https://huggingface.co/datasets/Babelscape/multinerd |
| Dialogue | DailyDialog | https://huggingface.co/datasets/roskoN/dailydialog |

Код авторов книги:

https://gitlab.com/tangibleai/nlpia2/

---

# Правила прохождения

## 1. Не читать слишком далеко вперёд

На каждую крупную идею:

```text
прочитал
 ↓
реализовал
 ↓
измерил
 ↓
посмотрел ошибки
 ↓
сделал вывод
```

Только потом двигаться дальше.

## 2. Всегда строить baseline

Перед сложной моделью должен существовать простой вариант.

Пример:

```text
TF-IDF + Logistic Regression
```

до:

```text
BERT
```

## 3. Не оценивать модель одной метрикой

Записывать:

- quality;
- latency;
- memory;
- количество параметров;
- стоимость inference;
- сложность deployment.

## 4. Делать Error Analysis

После каждого эксперимента смотреть реальные ошибки.

Не ограничиваться:

```text
Macro F1 = 0.87
```

Нужно понимать:

> Почему оставшиеся 13% ошибок происходят?

## 5. Сохранять результаты

Для каждого эксперимента фиксировать:

```text
model
dataset version
preprocessing
hyperparameters
metrics
latency
notes
```

---

# Порядок прохождения

```text
Глава 2
Tokenization / preprocessing
        ↓
Глава 3
TF-IDF + Logistic Regression
        ↓
Глава 4
SVD + semantic search
        ↓
Глава 5
MLP
        ↓
Глава 6
Word2Vec / embeddings
        ↓
STS-B
Semantic similarity
        ↓
Глава 7
TextCNN
        ↓
Глава 8
BiLSTM
        ↓
Глава 9
BERT / DistilBERT
        ↓
Глава 10
LLM zero-shot / few-shot
        ↓
SQuAD
Question Answering
        ↓
Глава 11
NER / MultiNERD
        ↓
Глава 12
Dialog system
        ↓
FastAPI + Docker
```

---

# Что делать прямо сейчас

Ты находишься на главе 2.

Пока не нужно трогать BERT, RAG, LoRA или fine-tuning LLM.

Текущая задача:

- [ ] Скачать BANKING77.
- [ ] Посмотреть train/test split.
- [ ] Посмотреть список классов.
- [ ] Выбрать сначала 10–15 классов.
- [ ] Посчитать длины текстов.
- [ ] Сравнить несколько способов токенизации.
- [ ] Построить vocabulary.
- [ ] Найти самые частые unigram.
- [ ] Найти самые частые bigram.
- [ ] Проверить lowercase.
- [ ] Проверить stopwords.
- [ ] Проверить stemming.
- [ ] Проверить lemmatization.
- [ ] Сравнить размер vocabulary.
- [ ] Записать выводы в README.
- [ ] После главы 3 построить TF-IDF + Logistic Regression baseline.

Главная цель главы 2:

> Научиться осознанно превращать сырой текст в представление, пригодное для дальнейшей обработки моделью.

Главная цель всего плана:

> На одной и той же задаче увидеть эволюцию NLP от токенизации и TF-IDF до Transformers и LLM и научиться выбирать инструмент под задачу, а не использовать LLM по умолчанию.
