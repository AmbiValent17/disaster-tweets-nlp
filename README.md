# NLP with Disaster Tweets

Бинарная классификация твитов: сообщение о реальном бедствии или нет.
Учебный проект по NLP

## Задача

Kaggle competition [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).
На вход подается текст твита, на выходе `target`:
- `1` если твит описывает реальное бедствие
- `0` если нет

Метрика на Kaggle: F1-score

## Данные

- `train.csv`
- `test.csv`
- Колонки: `id`, `keyword`, `location`, `text`, `target`

## Структура репозитория
.
├── 01_eda_and_baseline.ipynb # EDA, очистка, TF-IDF + LogReg
├── 02_transformers.ipynb # fine-tuning DistilBERT
├── requirements.txt
└── README.md

## Что было сделано

### EDA

- Заполнил пропуски в `keyword` значением `none`. Колонку `location` не использовал: 33% пропусков, значения грязные (свободный текст, сокращения).
- Нашел дубликаты по тексту. Часть из них с конфликтующими метками (один и тот же текст размечен и как 0, и как 1). Удалил все конфликтные строки, из остальных оставил по одному экземпляру каждого текста.
- Посмотрел топ-слова по классам. Для `target=1` характерны fire, disaster, police, crash, killed. Для `target=0` бытовая лексика: like, love, video, body.
- Длина твитов в словах у обоих классов почти одинаковая (медиана 15), поэтому длину как признак не использовал. `max_length=64` для токенизатора хватает с запасом: max 31 слово на твит.

### Baseline: TF-IDF + LogisticRegression

- Очистка текста: нижний регистр, удаление URL, упоминаний, HTML-entity (`&amp;`), пунктуации.
- TfidfVectorizer с `ngram_range=(1,2)`, `min_df=2`, `max_df=0.9`.
- 5-fold StratifiedKFold, метрика F1.
- С `class_weight='balanced'` F1 вырос с 0.740 до 0.760, разница значимая, оставил balanced.

Подбирал порог предсказания по F1. Оптимум оказался на 0.48, дефолтный 0.5 практически совпадает, поэтому оставил 0.5.

Заметил, что recall для target=1 ниже: модель пропускает около 29% реальных бедствий. Это ограничение bag-of-words: слова вроде fire в переносном смысле модель не отличает от настоящего сообщения о пожаре.

### DistilBERT

Fine-tuning `distilbert-base-uncased`, `max_length=64`.

В отличие от baseline, для трансформера применил только минимальную очистку (убрать `&amp;` и нормализовать пробелы)

Текущий результат: F1 = 0.8012 на val-выборке, на тестовой kaggle - 0.82868

## Результаты

| Модель | F1 |
|--------|-----|
| TF-IDF + LogisticRegression (5-fold CV) | 0.760 |
| DistilBERT (hold-out, в процессе тюнинга) | 0.777 |

Baseline считал на 5-fold CV, трансформер на одном hold-out split. Сравнение не идеально честное, для полноценного сравнения нужно гонять оба на одинаковых сплитах. Для пет-проекта оставил так.

## Стек

- Python 3.12
- pandas, numpy
- scikit-learn
- PyTorch
- HuggingFace Transformers
- matplotlib, seaborn