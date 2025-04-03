
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# =============================
# PLOTS FOR Q1-3: SPEAKER SEPARATION
# =============================
sep_path = "results/speaker_separation/evaluation_results.csv"
if os.path.exists(sep_path):
    df_sep = pd.read_csv(sep_path)
    os.makedirs("results/speaker_separation/plots", exist_ok=True)

    # KDE
    plt.figure(figsize=(10, 6))
    for metric in ['SDR', 'SIR', 'SAR', 'PESQ']:
        sns.kdeplot(df_sep[metric], label=metric, linewidth=2)
    plt.title("Q1-3: KDE Distribution of Enhancement Metrics")
    plt.xlabel("Metric Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("results/speaker_separation/plots/separation_kde_metrics.png")
    plt.close()

    # Boxplot
    melted = df_sep.melt(id_vars=["file", "speaker1", "speaker2"], value_vars=['SDR', 'SIR', 'SAR', 'PESQ'])
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=melted, x='variable', y='value', palette="coolwarm")
    plt.title("Q1-3: Boxplot of SDR, SIR, SAR, PESQ")
    plt.savefig("results/speaker_separation/plots/separation_metric_boxplot.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_sep[['SDR', 'SIR', 'SAR', 'PESQ']].corr(), annot=True, cmap="Purples", fmt=".2f")
    plt.title("Q1-3: Correlation Heatmap of Metrics")
    plt.savefig("results/speaker_separation/plots/separation_metric_correlation.png")
    plt.close()

# =============================
# PLOTS FOR Q1-4: ENHANCED PIPELINE
# =============================
enh_path = "results/enhanced_pipeline/evaluation_results.csv"
if os.path.exists(enh_path):
    df_enh = pd.read_csv(enh_path)
    os.makedirs("results/enhanced_pipeline/plots", exist_ok=True)

    # KDE
    plt.figure(figsize=(10, 6))
    for metric in ['SDR', 'SIR', 'SAR', 'PESQ']:
        sns.kdeplot(df_enh[metric], label=metric, linewidth=2)
    plt.title("Q1-4: KDE Distribution of Enhancement Metrics")
    plt.xlabel("Metric Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("results/enhanced_pipeline/plots/q14_kde_metrics.png")
    plt.close()

    # Boxplot
    melted = df_enh.melt(id_vars=["file", "speaker1", "speaker2"], value_vars=['SDR', 'SIR', 'SAR', 'PESQ'])
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=melted, x='variable', y='value', palette="magma")
    plt.title("Q1-4: Boxplot of SDR, SIR, SAR, PESQ")
    plt.savefig("results/enhanced_pipeline/plots/q14_boxplot_metrics.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_enh[['SDR', 'SIR', 'SAR', 'PESQ']].corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Q1-4: Correlation Heatmap of Metrics")
    plt.savefig("results/enhanced_pipeline/plots/q14_corr_heatmap.png")
    plt.close()
