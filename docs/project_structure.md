squares-in-squares/
├── pyproject.toml
├── .gitignore
├── STATE.md                      # статус, обновляется после каждого чата
├── docs/
│   └── metric_notes.md
├── src/sqin/
│   ├── __init__.py
│   ├── vendor/
│   │   ├── __init__.py
│   │   └── metric.py             # squares_in_square_metric.py БЕЗ правок
│   ├── io.py                     # чтение/запись сабмита, id = n(n-1)/2 + square_id, repr
│   ├── geometry.py               # углы, numba-SAT префильтр
│   ├── validate.py               # SEP_EPS=1e-12, гейт через vendor/metric.py
│   ├── store.py                  # best[n], строгое улучшение, push-down
│   ├── grid.py                   # тривиальная база
│   └── opt/                      # чаты 03-04: penalty, anneal, polish
├── scripts/
│   └── trivial_stats.py          # скрипт A
├── tests/
│   └── test_metric_assumptions.py  # скрипт B как pytest (T1-T4)
├── data/                         # в .gitignore
│   ├── best/                     # по одному файлу на n
│   ├── external/                 # публичные решения с источником и лицензией
│   └── submissions/
└── runs/                         # логи, в .gitignore

Правила, чтобы не было свалки:

- Вся логика живёт в src/sqin/. В scripts/ только тонкие запускалки.
- Метрику копируем в vendor/ один раз и не правим. Если Kaggle её обновит, перезаписываем файл целиком.
- Данные и CSV в git не попадают: data/, runs/ и *.csv в .gitignore.
- Каждый чат добавляет модули в свой слой, существующие файлы не переписывает.
