# DroPE: ограниченное воспроизведение на двух T4

Домашнее задание №2. Статья: **Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings**, Gelberg et al., Sakana AI / Oxford, 2025. [Статья](https://arxiv.org/abs/2512.12167), [код авторов](https://github.com/SakanaAI/DroPE).

## Материалы для проверки

- [Презентация](docs/presentation_drope.pptx)
- [Отчёт PDF](docs/report.pdf)
- [Исполненный аналитический ноутбук](notebooks/01_analysis.ipynb).
- `src/`: первый этап, продолжение, оценка и CPU-анализ.
- `results/`: сохранённые метрики; `figures/`: графики; `configs/`: историческая конфигурация и окружение.

**Результат:** частичное восстановление LM-качества после отключения RoPE. На 2048 test PPL снизилась с 612.54 до 30.95, но исходная модель имеет 12.64. На 8192 dev-tuned DroPE лучше немасштабированной RoPE (81.02 против 286.88), однако хуже YaRN (12.11) и Dynamic NTK (19.26). Retrieval не восстановлен: DroPE 0/12 на коротком NIAH и 0/24 на 2048; RoPE 10/12 и 15/24 соответственно. Это ограниченное воспроизведение процедуры, а не подтверждение всех выводов статьи.
