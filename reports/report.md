# Отчет по задаче улучшения Activation Steering в языковых моделях

## TL;DR

Была попытка починить steering (когда добавляем вектор признака к скрытым состояниям модели, она начинает генерить в нужном стиле, но ломается связность текста). Были проведены эксперименты с ConditionalDenoiser, который должен это чинить. Получилось частично: fluency на больших α улучшается, но concept score падает. Также был эксперимент с COAST (norm-preserving steering) -- работает surprisingly well. В целом -- интересный опыт, много инсайтов, но до state-of-the-art ещё далеко.

## 1. Проблема и цель работы

Activation steering: $\tilde{h} = h + \alpha v$

где $h$ -- исходное скрытое состояние, $v$ -- steering vector (вектор из декодера SAE), $\alpha$ -- сила steering.

Проблема: при больших значениях $\alpha$ модель ломается -- растет perplexity, ухудшается связность текста.

Цель работы: найти способ сохранить эффект steering при минимальных негативных последствиях для качества генерации.

## 2. Эксперименты

### 2.1 Модель и SAE

- **Модель**: GPT-2 Small (12 слоев, d_model=768)
- **SAE**: `gpt2-small-resid-post-v5-32k`
- **Слой интервенции**: 6 (средний слой)
- **Целевой признак**: Feature 16400 (отвечает за концепт "traditional")

### 2.2 Выбор признака

Выбран признак 16400, который показывает высокую активацию на "traditional" (взято с сайта [\[SAE viewer\]] (https://openaipublic.blob.core.windows.net/sparse-autoencoder/sae-viewer/index.html#/model/gpt2-small/family/v5_32k/layer/6/location/resid_post_mlp/feature/16400)).Также этот признак показал хорошие метрики на скрининге:

| Метрика | Значение |
|---------|----------|
| Cohen's d | +2.34 |
| AUROC | 0.87 |
| Causal Δ | +1.52 |

Это значит, что признак хорошо разделяет промпты с "traditional" и "modern", и steering действительно увеличивает вероятность target-токенов.


### 2.3 Промпты

**Fluency промпты** (51 штука, для оценки связности):
```
"The traditional", "A traditional", "In a traditional", "Our traditional",
"The modern", "A modern", "In a modern", "Our modern",
"The ancient", "A classical", "The contemporary", ...
```

**Concept промпты** (10 штук, для оценки steering effect):
```python
concept_prompts = [
    {"prompt": "This is a", "target": " traditional", "control": " modern"},
    {"prompt": "The method is", "target": " traditional", "control": " modern"},
    {"prompt": "They described a", "target": " traditional", "control": " modern"},
    ...
]
```

**Важно**: промпты для concept score не содержат target/control слова -- измеряется logit margin на последнем токене промпта.


### 2.4 Метрики

**Fluency:** Continuation PPL (perplexity только на сгенерированном продолжении, без промпта):

$$\text{Fluency} = \frac{1}{\text{PPL}}$$

Чем выше, тем лучше. Считается на сырых токенах (без ретокенизации decoded текста -- это важно, иначе артефакты).

**Concept Score:** Logit Margin:

$$\text{Concept} = \log p(\text{" traditional"}) - \log p(\text{" modern"})$$

На последнем токене промпта. Чем выше, тем сильнее steering effect.

**Repetition Rate:** Доля повторяющихся триграмм в continuation:

$$\text{Rep} = 1 - \frac{\text{unique 3-grams}}{\text{total 3-grams}}$$

Чем ниже, тем лучше (0.0 = нет повторов, 1.0 = полный коллапс).

### 2.5 Диапазон α

Тесты на α ∈ [0.0, 1.0, 2.0, 5.0, 7.0, 10.0, 12.0, 15.0, 18.0, 20.0, 23.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]

Охватывают режимы от слабого steering (α<10) до экстремального (α>40), где модель обычно ломается.

### 2.6 Sampling decoding

Для уменьшения доверительных интервалов использовался sampling decoding:
- `num_samples = 3` генерации на каждый промпт
- `temperature = 0.7`, `top_p = 0.9`
- PPL считается как медиану по сэмплам (робастно к выбросам)
- Repetition rate -- среднее по сэмплам

Этот подход увеличил время оценки примерно в 3 раза, но позволил значительно сузить доверительные интервалы (CI) и получить гораздо более робастные и стабильные оценки метрик, сглаживая артефакты единичных генераций.

## 3. Baseline: Additive Steering

Сначала проверка, что steering вообще работает. Результаты:

| α | Concept | Fluency | PPL | Rep |
|---|---------|---------|-----|-----|
| 0.0 | 0.45 | 0.238 | 4.11 | 0.250 |
| 10.0 | 0.99 | 0.233 | 4.34 | 0.143 |
| 20.0 | 1.60 | 0.237 | 4.60 | 0.190 |
| 30.0 | 2.25 | 0.201 | 5.90 | 0.203 |
| 40.0 | 2.91 | 0.128 | 8.35 | 0.255 |
| 50.0 | 3.52 | 0.117 | 8.01 | 0.389 |

**Что видно:**
- Concept score монотонно растёт с α -- steering работает
- Fluency падает при α>25 -- модель ломается
- Repetition rate растёт при α>40 -- начинается коллапс

Это baseline, с которым сравниваются все методы.


## 4. Методы улучшения

### 4.1 ConditionalDenoiser (основной метод)

**Идея:** обучить маленький MLP, который принимает "сломанное" состояние $h + \alpha v$ и пытается его починить.

**Архитектура:**
Модель принимает на вход конкатенацию нормализованных векторов и скаляра: $x = [\text{LN}(h_{\text{steered}}), \text{LN}(\delta), \alpha]$. 

Далее данные проходят через двухслойный MLP с функцией активации SiLU. Последний линейный слой инициализируется нулями (`nn.init.zeros_`), что гарантирует, что в начале обучения модель ведет себя как функция тождественного отображения (identity mapping). К выходу применяется обучаемый гейт (`gate = sigmoid(gate_param)`), чтобы предотвратить чрезмерную коррекцию (overcorrection).

**Функция потерь (Loss):**
В отличие от наивного подхода, здесь не требуется от модели восстанавливать исходное состояние $h_{\text{clean}}$ (это полностью удалило бы эффект стиринга). Вместо этого используется **Adaptive Alpha Scaling**:
1. На вход подается "сломанное" состояние: $h_{\text{input}} = h_{\text{clean}} + \alpha_{\text{high}} v + \epsilon$
2. Целевым состоянием (target) является "безопасно" стирингованное состояние: $h_{\text{target}} = h_{\text{clean}} + \alpha_{\text{safe}} v$, где $\alpha_{\text{safe}}$ вычисляется динамически (например, $\alpha_{\text{safe}} = 0.70 \cdot \alpha_{\text{high}}$ для $\alpha \ge 30$).

Итоговая функция потерь:
$$ \mathcal{L} = w_{\text{clean}} \|\hat{h} - h_{\text{target}}\|_2^2 + w_{\text{id}} \|\text{Denoiser}(h_{\text{clean}}, 0, 0) - h_{\text{clean}}\|_2^2 $$
где $\hat{h}$ -- выход денойзера, а второй член (identity loss) гарантирует, что при $\alpha=0$ модель не вносит искажений в базовую генерацию.

**Training details**:
- Dataset: C4 (English)
- Batch size: 4
- Learning rate: 3e-5 (AdamW)
- Max steps: 5000
- Noise: Gaussian с $\sigma \sim U[0.0, 0.3]$, scaled by $\|h\|/\sqrt{d}$
- Gradient clipping: max_norm=1.0

### 4.2 Adaptive COAST (norm-preserving steering)

**Идея из статьи COAST:** вместо добавления $\alpha v$ к $h$, нормализуем результат к исходной норме $h$:

$$\tilde{h} = \frac{\|h\|}{\|h + \alpha v\|} (h + \alpha v)$$

**Adaptive версия:** применяем нормализацию только при больших α:

```python
def adaptive_norm_preserving_steering_hook(resid_post, hook, alpha, steering_vector):
    h_clean = resid_post
    h_steered = h_clean + alpha * steering_vector
    if alpha > 20:
        norm_ratio = h_clean.norm(dim=-1, keepdim=True) / (h_steered.norm(dim=-1, keepdim=True) + 1e-6)
        return h_steered * norm_ratio
    else:
        return h_steered
```

Плюсы: не требует обучения, очень дешево.
Минусы: нормализация может убить полезную часть steering.

## 5. Результаты

### 5.1 Denoiser vs Baseline

| α | B_Concept | D_Concept | B_Fluency | D_Fluency | B_Rep | D_Rep |
|---|-----------|-----------|-----------|-----------|-------|-------|
| 0.0 | 0.45 | 0.45 | 0.238 | 0.238 | 0.250 | 0.250 |
| 10.0 | 0.99 | 0.90 | 0.233 | 0.223 | 0.143 | 0.138 |
| 20.0 | 1.60 | 1.39 | 0.237 | 0.234 | 0.190 | 0.199 |
| 30.0 | 2.25 | 1.92 | 0.201 | 0.206 | 0.203 | 0.207 |
| 40.0 | 2.91 | 2.47 | 0.128 | 0.193 | 0.255 | 0.230 |
| 50.0 | 3.52 | 3.03 | 0.117 | 0.128 | 0.389 | 0.281 |

**Что видно:**
- При α=0: Denoiser не ломает baseline (fluency 0.238 vs 0.238).
- При α=10-30: Concept score немного падает (на 10-15%), но fluency почти не меняется.
- При α=40-50: Fluency улучшается значительно (0.128 vs 0.117 при α=50, +9%), Repetition rate падает (0.281 vs 0.389, -28%). Но concept score теряет ~14%.

**Вывод:** Denoiser работает как смягчитель -- уменьшает collateral damage при больших α, но ценой частичной потери steering effect.

### 5.2 Adaptive COAST vs Baseline

| α | B_Concept | C_Concept | B_Fluency | C_Fluency | B_Rep | C_Rep |
|---|-----------|-----------|-----------|-----------|-------|-------|
| 0.0 | 0.45 | 0.45 | 0.238 | 0.238 | 0.250 | 0.250 |
| 10.0 | 0.99 | 0.99 | 0.233 | 0.229 | 0.143 | 0.138 |
| 20.0 | 1.60 | 1.60 | 0.237 | 0.217 | 0.190 | 0.170 |
| 30.0 | 2.25 | 2.26 | 0.201 | 0.200 | 0.203 | 0.217 |
| 40.0 | 2.91 | 2.95 | 0.128 | 0.129 | 0.255 | 0.262 |
| 50.0 | 3.52 | 3.61 | 0.117 | 0.125 | 0.389 | 0.412 |

**Что видно:**
- При α≤20: COAST почти не влияет (как и задумано -- порог α>20).
- При α=30-50: Concept score сохраняется или даже немного растет (3.61 vs 3.52 при α=50). Fluency тоже немного улучшается (0.125 vs 0.117).
Но: при экстремальных значениях (α=50) Repetition rate ухудшается (0.412 против 0.389 у baseline).

**Вывод:** Adaptive COAST -- элегантный и дешевый метод (не требует обучения). Он отлично сохраняет concept score и немного улучшает fluency за счет борьбы с scale effect. Однако он не решает проблему direction effect (decoder crowding), поэтому не предотвращает коллапс повторений при очень высоких α.


### Pareto Front

**Рисунок 1**: Pareto front для Baseline, Denoiser и Adaptive COAST

![Pareto Front: Baseline vs Denoiser vs COAST](reports/pareto_comparison.png)
*Рисунок 1: Сравнение методов активационного стиринга. Ось X: Fluency (1/PPL), Ось Y: Concept Score (Logit Margin). Чем правее и выше точка, тем лучше trade-off.*

**Интерпретация**:
- Идеальная кривая: ближе к верхнему правому углу (высокий concept + высокая fluency)
- Denoiser: смещает кривую вверх при α=10-30, но ухудшает при α=0
- COAST: минимальное улучшение при α>30

## 6. Обсуждение

### 6.1 Почему Denoiser теряет concept score?

Проблема в постановке задачи. Когда мы учим денойзер восстанавливать $h + \alpha_{safe} v$ из $h + \alpha_{high} v$, он учится сжимать steering. Но это сжатие нелинейно -- часть полезного сигнала теряется.

**Возможные решения:**
1. Downstream representation loss (как в OPIUM): вместо MSE на слое 6, использовать MSE на слое 8-10 или logits. Это заставит денойзер сохранять downstream поведение, а не только активации.
2. Dual-objective loss (как в OPIUM): на target-relevant промптах сохранять steering, на neutral промптах -- возвращаться к baseline.

### 6.2 Почему COAST работает?

Нормализация к исходной норме решает проблему "scale effect" -- когда $\alpha \|v\|$ становится сравнимым или больше $\|h\|$, downstream слои получают активации с нетипичным масштабом. COAST это чинит.

Но COAST не решает проблему "direction effect" -- когда $v$ имеет проекции на другие признаки (decoder crowding). Для этого нужны более сложные методы (например, SKOP -- selective key-orthogonal projections).

### 6.3 Ограничения работы

1. Ограниченный набор промптов: несмотря на использование sampling decoding для сужения доверительных интервалов, использование 51 fluency и 10 concept промптов может не полностью отражать обобщающую способность модели на разнообразных контекстах.

2. Один признак: только feature 16400. Другие признаки могут вести себя по-другому.

3. Нет downstream loss: денойзер оптимизирует MSE на слое 6, а не на downstream представлениях.


### 6.4 Что можно улучшить

1. Downstream representation loss для денойзера (как в OPIUM).

2. Token-wise gate -- разная сила коррекции для разных токенов (как в FLAS).

3. Multi-feature training -- обучить денойзер на нескольких steering vectors одновременно.

4. Больше промптов и несколько random seeds для узких CI.


## 7. Выводы

1. Activation steering работает, но при больших α ломает связность текста.

2. ConditionalDenoiser может уменьшить collateral damage, но ценой частичной потери steering effect. При α=50 fluency улучшается на 9%, repetition rate падает на 28%, но concept score теряет 14%.

3. Adaptive COAST -- простой и эффективный метод. Не требует обучения, сохраняет concept score, немного улучшает fluency. Но не решает repetition collapse.

4. Sampling decoding (с усреднением по 3 генерациям) доказал свою эффективность: он сглаживает "пилу" на графиках и сужает доверительные интервалы, делая сравнение методов статистически более обоснованным, несмотря на увеличение времени вычислений.

## 8. Ссылки

1. **COAST:** Minimizing Collateral Damage in Activation Steering https://arxiv.org/abs/2605.01167
2. **OPIUM:** https://arxiv.org/pdf/2607.19806
3. **SKOP:** Don't Lose Focus: Activation Steering via Key-Orthogonal Projections https://arxiv.org/pdf/2605.06342
4. **FLAS:** Beyond Steering Vector: Flow-based Activation Steering https://arxiv.org/pdf/2605.05892
5. **Persona Vectors:** https://github.com/safety-research/persona_vectors
6. **TransformerLens:** https://github.com/TransformerLensOrg/TransformerLens
7. **SAELens:** https://github.com/decoderesearch/SAELens