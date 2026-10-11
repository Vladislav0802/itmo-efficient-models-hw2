# Веса и повторная оценка

Финальный checkpoint: **step 18689**, промежуточный: **step 8712**. Base model доступен публично: https://huggingface.co/HuggingFaceTB/SmolLM-360M, revision `59f7ef243ee09a72cbc14cb054393a3e3b771d41`.

Обученные веса находятся в сохранённом Kaggle Output/локальном резервном архиве пользователя. Публичная ссылка на них не предоставлена, поэтому здесь не заявляется возможность скачать финальные веса по ссылке. Полный путь определяет `checkpoint_metadata.json`: `hf/<full_resume_tag>/`. Это HF directory с config.json, model*.safetensors и файлами токенизатора. Не публикуйте многогигабайтные веса обычным Git.

Для run_check/run_extra к весам нужны `config.json`, `checkpoint_metadata.json`, `cache/test.npy`, `cache/dev.npy` и `code/custom_models/{drope.py,attention.py}` из того же Output. Для продолжения дополнительно нужны полный `resume_state/`, `cache/manifest.json`, train caches, `training_result.json` и состояния потоков данных.

Если требуется предоставить преподавателю именно уже обученный checkpoint, опубликуйте этот минимальный **evaluation bundle** как Kaggle Dataset или HF repo и добавьте ссылку в README. Пока эта ссылка отсутствует, доступны код повторного обучения и полностью воспроизводимый CPU-анализ метрик; независимая повторная оценка ваших финальных весов требует их передачи.

Не удаляйте единственную резервную копию полного исходного Output. Удаление optimizer из копии для Git допустимо; из единственного resume-архива — нет.
