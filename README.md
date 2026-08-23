# Steering Denoiser: Mitigating Collateral Damage in Activation Steering

Этот репозиторий содержит код и эксперименты по улучшению activation steering в языковых моделях. Предлагается использовать Conditional Denoiser с адаптивным масштабированием для уменьшения "колатерального ущерба" (роста перплексии и деградации связности текста) при сильном стиринге.

## Проблема
Стандартный activation steering ($\tilde{h} = h + \alpha v$) позволяет управлять поведением LLM, но при больших $\alpha$ модель ломается: резко растет perplexity, появляются повторы и теряется связность текста.

## Решение
Вместо того чтобы обучать модель слепо возвращать активации к исходному состоянию $h$ (что полностью удаляет эффект стиринга), используется Selective Alpha Scaling. Денойзер учится сжимать чрезмерное возмущение $\alpha_{high} v$ до безопасного уровня $\alpha_{safe} v$, сохраняя целевой концепт, но убирая разрушительные артефакты.

Также в репозитории реализован baseline-метод **COAST** (norm-preserving steering) для сравнения.

## Установка

Рекомендуется использовать `uv`:

```
uv venv --python 3.11
source .venv/bin/activate  # или .venv\Scripts\activate на Windows
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv sync
```

## Использование
Все эксперименты управляются через Hydra-конфиги (configs/).

1. Обучение Denoiser
```
python src/train.py
```
Чекпоинт и логи будут сохранены в папку outputs/YYYY-MM-DD/HH-MM-SS/.
2. Оценка и построение графиков
Для сравнения Baseline, Denoiser и COAST использовать скрипт оценки. Указать путь к чекпоинту:
```
python src/evaluate.py evaluate.mode="all" evaluate.checkpoint_path="outputs/202X-XX-XX/XX-XX-XX/checkpoints/denoiser.pt"
```
Доступные режимы (evaluate.mode):

* "baseline": только обычный additive steering
* "denoiser": baseline + обученный denoiser
* "coast": baseline + norm-preserving steering
* "all": сравнение всех трех методов на одном графике

## Основные результаты
На задаче усиления концепта "traditional" (GPT-2 Small, SAE feature 16400):
- При α=40: Denoiser кардинально улучшает связность: Fluency растет с 0.126 до 0.193 (+53%), а PPL падает с 8.35 до 5.68. Concept Score сохраняется на уровне 2.47 (против 2.91 у baseline).
- При α=50: Denoiser снижает Repetition Rate с 0.39 до 0.28 (-28%) и улучшает Fluency с 0.117 до 0.128, сохраняя при этом ~86% Concept Score (3.03 против 3.52 у baseline).
- При α=0: Полное совпадение с Baseline (Identity preservation работает идеально).

COAST показывает отличные результаты без обучения, полностью сохраняя Concept Score и незначительно улучшая Fluency, но хуже справляется с repetition collapse на экстремальных α.

## Структура проекта

* configs/: Hydra-конфигурации для данных, модели и оценки.
* src/: Исходный код (train.py, evaluate.py, models.py, steering.py).
* reports/: Отчет с результатами.

## Checkpoints
[\[HuggingFace\]](https://huggingface.co/ann-chern/steering_denoiser/tree/main/checkpoints)

## Ссылки

1. **COAST:** Minimizing Collateral Damage in Activation Steering — https://arxiv.org/abs/2605.01167
2. **TransformerLens:** https://github.com/TransformerLensOrg/TransformerLens
3. **SAELens:** https://github.com/decoderesearch/SAELens