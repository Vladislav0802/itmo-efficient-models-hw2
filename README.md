# DroPE: ограниченное воспроизведение на двух T4

Домашнее задание №2. Статья: **Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings**, Gelberg et al., Sakana AI / Oxford, 2025. [Статья](https://arxiv.org/abs/2512.12167), [код авторов](https://github.com/SakanaAI/DroPE).

## Материалы для проверки

- [Презентация](docs/presentation.pptx), [PDF презентации](docs/presentation.pdf).
- [Отчёт PDF](docs/report.pdf), [текст отчёта](docs/report.md).
- [Исполненный аналитический ноутбук](notebooks/01_analysis.ipynb).
- `src/`: первый этап, продолжение, оценка и CPU-анализ.
- `results/`: сохранённые метрики; `figures/`: графики; `configs/`: историческая конфигурация и окружение.

**Результат:** частичное восстановление LM-качества после отключения RoPE. На 2048 test PPL снизилась с 612.54 до 30.95, но исходная модель имеет 12.64. На 8192 dev-tuned DroPE лучше немасштабированной RoPE (81.02 против 286.88), однако хуже YaRN (12.11) и Dynamic NTK (19.26). Retrieval не восстановлен: DroPE 0/12 на коротком NIAH и 0/24 на 2048; RoPE 10/12 и 15/24 соответственно. Это ограниченное воспроизведение процедуры, а не подтверждение всех выводов статьи.

## Анализ без GPU и весов

Python 3.9–3.13; проверено локальным выполнением всех ячеек. Из корня проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m ipykernel install --user --name drope-analysis --display-name "DroPE analysis"
python src/analyze_results.py
```

В VS Code открыть `notebooks/01_analysis.ipynb`, выбрать **DroPE analysis** и Run All. `numpy` из другого Python не доступен этому ядру. Предварительно выполненные результаты уже встроены в ноутбук.

## Обучение с нуля

Linux, NVIDIA CUDA, **2 GPU по 16 GB**. Исходный запуск: T4 ×2, PyTorch 2.11.0+cu128, Python 3.13; полный freeze сохранён в `configs/environment_pip_freeze.txt`. Эти версии описывают фактическую сессию, а не универсальную установку для любого драйвера. Сначала установить подходящий CUDA PyTorch, затем:

```bash
python -m pip install -r requirements-gpu.txt
python src/launch_stage1.py --lr 3e-5 --max-steps 8712 --hours 11 --output-root runs/stage1
```

Подготовка данных может занять 15–30 минут, шаг около 3–3.5 с. Параметр hours ограничивает бюджет; остановка по времени может дать меньше шагов. Скрипт скачивает base checkpoint и датасет на фиксированных revisions, получает два авторских модуля на фиксированном commit и сверяет их байты. Сохраняет HF-веса и **полное состояние optimizer/RNG/stream** каждые 250 шагов и в конце. Для продолжения выбрать каталог с `config.json` и `checkpoint_metadata.json`:

```bash
python src/launch.py --input /ABS/PATH/TO/STAGE1_RUN --max-new-steps 9977 --hours 11 --eval-minutes 55 --output-root runs/stage2
```

Нужен **полный исходный Output**, включая `hf/`, `resume_state/`, `cache/`, `code/custom_models/`. Очищенная папка проекта не является resume-checkpoint. Требуется минимум 12 GiB свободного рабочего диска, плюс место для исходного Input.

## Оценка сохранённых весов

```bash
python src/evaluation/run_check.py --input /ABS/PATH/TO/STAGE2_RUN --minutes 55 --output-root runs/check
python src/evaluation/run_extra.py --input /ABS/PATH/TO/STAGE2_RUN --minutes 45 --output-root runs/extra
```

Две независимые роли используют GPU0 (RoPE) и GPU1 (DroPE). Первая оценка включает LM, dev-подбор beta, контроль KV-cache и малый NIAH; дополнительная — расширенный NIAH и LM-базовые YaRN/Dynamic NTK. В новой версии исправлены метаданные YaRN. Исторические числа после этой правки не пересчитывались.

На Kaggle включить Internet и GPU T4 ×2. Запускать из корня загруженного проекта; для сохранённой версии **Save Version → Save & Run All** запускает все ячейки самостоятельно. Перед закрытием проверить статус **Complete**, затем наличие файлов во вкладке Output. Для следующего запуска подключить Output этой версии через Add Input. Сохранение только текста notebook не сохраняет интерактивные веса.

## Данные, веса, происхождение чисел

Метрики и анализ доступны без весов. Веса не входят в Git: см. [WEIGHTS](docs/WEIGHTS.md). Дополнительный архив был потерян после разрыва Kaggle-сессии; `results/extra/` честно восстановлен из консольного вывода. Никакие генерации или prompt hashes не выдуманы: [provenance](results/extra/PROVENANCE.md).

Новые launch-обёртки и дополнительный evaluator проверены на синтаксис и согласованность интерфейсов; новый GPU-запуск в рамках подготовки сдачи не выполнялся. Сохранённые эксперименты выполнены историческими версиями, их конфигурации сохранены. Авторские модули и их дубликаты сверены: [атрибуция](THIRD_PARTY_NOTICES.md).
