# Происхождение стороннего кода

`src/custom_models/{attention.py,drope.py}` и байт-в-байт совпадающие копии в `src/evaluation/custom_models/` взяты из SakanaAI/DroPE, commit `260a8bb9f4fda0673266ddfee7668915aade8b33`:
https://github.com/SakanaAI/DroPE/tree/260a8bb9f4fda0673266ddfee7668915aade8b33/custom_models

Авторы статьи: Yoav Gelberg, Koshi Eguchi, Takuya Akiba, Edoardo Cetin. Копии оставлены для самостоятельного запуска evaluation subprocess; это не два разных варианта архитектуры. Адаптер совместимости KV-cache расположен в нашем `evaluation/worker.py`, а не в авторских файлах.

В дереве указанного upstream commit не обнаружен LICENSE/NOTICE; репозиторию не приписывается выдуманная MIT/Apache лицензия. Условия использования base-модели и датасета следует смотреть в их model/dataset cards. Наши экспериментальные скрипты и анализ документируют учебное воспроизведение.
