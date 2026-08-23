import matplotlib.pyplot as plt
import os
import numpy as np


def bootstrap_ci(values, n_bootstrap=100, confidence=0.95, statistic='median'):
    """Считает statistic (mean/median) и 95% CI через bootstrap."""
    values = np.array(values)
    n = len(values)
    stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        if statistic == 'median':
            stats.append(np.median(sample))
        else:
            stats.append(np.mean(sample))

    stat_value = np.median(stats) if statistic == 'median' else np.mean(stats)
    ci_low = np.percentile(stats, (1 - confidence) / 2 * 100)
    ci_high = np.percentile(stats, (1 + confidence) / 2 * 100)
    return stat_value, ci_low, ci_high


def plot_pareto_comparison(results_dict, out_dir, title=None):
    """
    Строит сравнительный график Pareto front для любого количества режимов.

    Args:
        results_dict: dict с ключами режимов (baseline, denoiser, coast, slerp, ...)
        out_dir: директория для сохранения графика
        title: заголовок графика (опционально)
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    all_alphas = []
    for mode, results in results_dict.items():
        if results:
            all_alphas.extend([r["alpha"] for r in results])

    if not all_alphas:
        print("No results to plot!")
        return

    vmin = min(all_alphas)
    vmax = max(all_alphas)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    mode_colors = {
        'baseline': 'black',
        'denoiser': 'red',
        'coast': 'green',
        'slerp': 'blue',
        'default': 'purple'
    }

    def plot_series(results, mode):
        if not results:
            return None

        if mode in mode_colors:
            color = mode_colors[mode]
        else:
            color = mode_colors['default']

        alphas = [r["alpha"] for r in results]
        sorted_idx = np.argsort(alphas)
        alphas_sorted = [alphas[i] for i in sorted_idx]

        fluency_means, fluency_ci_low, fluency_ci_high = [], [], []
        concept_means, concept_ci_low, concept_ci_high = [], [], []

        for r in results:
            ppl_list = r.get("ppls_per_prompt", [r["ppl"]])
            fluency_samples = [1.0 / p for p in ppl_list]
            f_mean, f_low, f_high = bootstrap_ci(fluency_samples, statistic='median')
            fluency_means.append(f_mean)
            fluency_ci_low.append(f_low)
            fluency_ci_high.append(f_high)

            margin_list = r.get("margins_per_prompt", [r["concept"]])
            c_mean, c_low, c_high = bootstrap_ci(margin_list, statistic='mean')
            concept_means.append(c_mean)
            concept_ci_low.append(c_low)
            concept_ci_high.append(c_high)

        s_idx = np.argsort(alphas_sorted)
        f_sorted = [fluency_means[i] for i in s_idx]
        c_sorted = [concept_means[i] for i in s_idx]
        f_low_sorted = [fluency_ci_low[i] for i in s_idx]
        f_high_sorted = [fluency_ci_high[i] for i in s_idx]

        ax.fill_betweenx(
            c_sorted, f_low_sorted, f_high_sorted,
            color=color, alpha=0.25, zorder=1
        )
        ax.plot(f_sorted, c_sorted, color=color, linewidth=2, alpha=0.8, label=mode.capitalize(), zorder=2)

        scatter = ax.scatter(
            fluency_means, concept_means,
            c=alphas_sorted, cmap=cmap, norm=norm,
            s=120, edgecolors='black', linewidths=1.2, zorder=3, marker='o'
        )

        step = 2 if len(alphas_sorted) <= 10 else 3
        for i, alpha in enumerate(alphas_sorted):
            if i % step == 0:
                ax.annotate(
                    f'α={alpha}',
                    (fluency_means[i], concept_means[i]),
                    textcoords="offset points",
                    xytext=(8, 5),
                    fontsize=8,
                    color='black',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')
                )

        return scatter

    for mode, results in results_dict.items():
        plot_series(results, mode)

    ax.set_xlabel("Fluency (1 / Continuation PPL) ↑", fontsize=13)
    ax.set_ylabel("Concept Score (Logit Margin) ↑", fontsize=13)

    if title:
        ax.set_title(title, fontsize=15, fontweight='bold')
    else:
        ax.set_title("Pareto Front Comparison (shaded = 95% CI)", fontsize=15, fontweight='bold')

    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046,
        pad=0.04
    )
    cbar.set_label('Steering Strength (α)', fontsize=12, rotation=270, labelpad=20)

    plt.tight_layout()

    plot_path = os.path.join(out_dir, "pareto_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")
