# Применение пакета для сдачи

Этот архив содержит только новые и изменённые файлы относительно drope_project(1).zip. Он не заменяет весь проект. Команды выполняются из корня проекта, где находятся src/, results/, configs/ и notebooks/.

```bash
unzip -o "$HOME/Downloads/DroPE_submission_patch.zip" -d .
rm -f .DS_Store __huggingface_repos__.json 01_analysis.ipynb
```

Удаляются только macOS-мусор, служебные метаданные импорта HF и дубликат notebook в корне. Основной ноутбук — notebooks/01_analysis.ipynb. Сохранённые results/, configs/, исходные custom_models/ и графики 01–06 оставьте: это экспериментальная доказательная база, они малы. Веса, optimizer, train caches, большие tar/zip и окружение в Git не добавлять; новая .gitignore исключает их. Исходную резервную копию Output сохранить отдельно.

Проверьте README и docs/: presentation.pptx, presentation.pdf, report.pdf, report.md. Презентация — 14 слайдов; отчёт — 5 страниц. Добавьте своё имя на титульный слайд при необходимости. Если правите PPTX, обновите presentation.pdf, чтобы версии совпадали.

CPU-проверка при установленном окружении:

```bash
python src/analyze_results.py
```

Для сдачи: создать GitHub repository или папку в существующем, добавить эти файлы и передать ссылку. Из уже настроенного Git-репозитория:

```bash
git add .
git status --short
git diff --cached --stat
git commit -m "Add DroPE reproduction report, presentation and runnable experiment"
git push
```

Если папка ещё не является Git-репозиторием, сначала создайте/подключите его обычным способом. В staged files не должно быть safetensors/pt/npy, .venv и больших архивов. PDF/PPTX, метрики и notebook должны быть staged.

Публичная ссылка на ваши финальные веса пока не предоставлена. Для полной независимой оценки желательно опубликовать evaluation bundle по docs/WEIGHTS.md и добавить ссылку. Сдавать только patch ZIP не нужно: сдаётся итоговая GitHub-папка с исходными файлами плюс этим дополнением.
