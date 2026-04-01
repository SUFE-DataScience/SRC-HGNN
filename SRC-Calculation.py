import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from pathlib import Path

mpl.rcParams["savefig.format"] = "pdf"

mpl.rcParams.update({
    "font.size": 16,
    "font.family": "Times New Roman",
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "legend.title_fontsize": 16,
    "figure.titlesize": 18,
})

OUT_DIR = Path("figures_pdf_png")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    data_file = Path("data.xlsx")
    df = pd.read_excel(data_file)

    required_cols = {"Date", "Label", "Rating", "Sentiment"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    if pd.api.types.is_numeric_dtype(df["Label"]):
        df["Label"] = df["Label"].map({0: "real", 1: "fake"})

    df["rating_norm"] = (df["Rating"] - 1) / 4.0
    df["SRC"] = 1 - np.abs(df["Sentiment"] - df["rating_norm"])

    df["date"] = df["Date"].dt.floor("D")
    daily_stats = df.groupby(["date", "Label"])["SRC"].agg(["mean", "std"]).reset_index()
    daily_stats["cv"] = daily_stats.apply(
        lambda row: row["std"] / row["mean"] if row["mean"] != 0 else 0.0,
        axis=1
    )
    daily_stats.rename(columns={"mean": "mean_SRC", "std": "std_SRC"}, inplace=True)

    year_ticks = pd.to_datetime(["2004", "2006", "2008", "2010", "2012", "2014"])
    start_date = daily_stats["date"].min()
    end_date = pd.to_datetime(f"{daily_stats['date'].max().year}-12-31")

    fig_a = plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=daily_stats,
        x="date",
        y="mean_SRC",
        hue="Label",
        palette={"real": "#e41a1c", "fake": "#377eb8"},
        style="Label",
        s=30,
        alpha=0.8,
        edgecolor="face"
    )
    plt.xlabel("Date")
    plt.ylabel("Mean SRC")
    plt.xlim(start_date, end_date)
    plt.ylim(0, 1)
    plt.xticks(year_ticks, labels=year_ticks.strftime("%Y"), rotation=45)
    plt.legend(title="Label", loc="lower right")
    plt.grid(True)
    fig_a.savefig(OUT_DIR / "A_mean_src_scatter.pdf", bbox_inches="tight")
    fig_a.savefig(OUT_DIR / "A_mean_src_scatter.png", bbox_inches="tight", dpi=600)
    plt.close(fig_a)

    fig_b = plt.figure(figsize=(14, 7))
    real_data = daily_stats[daily_stats["Label"] == "real"]
    fake_data = daily_stats[daily_stats["Label"] == "fake"]
    cv_ticks = np.arange(0, 1.6, 0.3)

    ax1 = plt.subplot(1, 2, 1)
    sc1 = ax1.scatter(
        real_data["date"],
        real_data["std_SRC"],
        c=real_data["cv"],
        cmap="viridis",
        s=(real_data["cv"] + 0.1) * 140,
        alpha=0.6,
        edgecolors="face",
        vmin=0, vmax=1.5
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Std SRC")
    ax1.set_xlim(start_date, end_date)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(year_ticks)
    ax1.set_xticklabels(year_ticks.strftime("%Y"), rotation=45)
    cbar_real = plt.colorbar(sc1, ax=ax1)
    cbar_real.set_label("CV (Coefficient of Variation)")
    cbar_real.set_ticks(cv_ticks)
    ax1.grid(True)
    ax1.text(0.5, -0.22, "(a) Real", transform=ax1.transAxes,
             ha="center", va="top", fontsize=17)

    ax2 = plt.subplot(1, 2, 2)
    sc2 = ax2.scatter(
        fake_data["date"],
        fake_data["std_SRC"],
        c=fake_data["cv"],
        cmap="viridis",
        s=(fake_data["cv"] + 0.1) * 140,
        alpha=0.6,
        edgecolors="face",
        vmin=0, vmax=1.5
    )
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Std SRC")
    ax2.set_xlim(start_date, end_date)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(year_ticks)
    ax2.set_xticklabels(year_ticks.strftime("%Y"), rotation=45)
    cbar_fake = plt.colorbar(sc2, ax=ax2)
    cbar_fake.set_label("CV (Coefficient of Variation)")
    cbar_fake.set_ticks(cv_ticks)
    ax2.grid(True)
    ax2.text(0.5, -0.22, "(b) Fake", transform=ax2.transAxes,
             ha="center", va="top", fontsize=17)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    fig_b.savefig(OUT_DIR / "B_std_cv_split.pdf", bbox_inches="tight")
    fig_b.savefig(OUT_DIR / "B_std_cv_split.png", bbox_inches="tight", dpi=600)
    plt.close(fig_b)

    daily_stats_rating = df.groupby(["date", "Label", "Rating"])["SRC"].agg(["mean", "std"]).reset_index()
    daily_stats_rating["cv"] = daily_stats_rating.apply(
        lambda row: row["std"] / row["mean"] if row["mean"] != 0 else 0.0,
        axis=1
    )
    daily_stats_rating.rename(columns={"mean": "mean_SRC", "std": "std_SRC"}, inplace=True)

    color_map = {"fake": "#377eb8", "real": "#e41a1c"}

    fig_c, axes_c = plt.subplots(nrows=2, ncols=5, figsize=(16, 6), sharex=True, sharey=True)
    for i, label in enumerate(["fake", "real"]):
        for j, rating in enumerate([1, 2, 3, 4, 5]):
            ax = axes_c[i, j]
            subset = daily_stats_rating[
                (daily_stats_rating["Label"] == label) &
                (daily_stats_rating["Rating"] == rating)
            ]
            ax.scatter(
                subset["date"],
                subset["mean_SRC"],
                s=30,
                alpha=0.8,
                color=color_map[label]
            )
            ax.set_xlim(start_date, end_date)
            ax.set_ylim(0, 1)
            ax.set_xticks(year_ticks)
            ax.set_xticklabels(year_ticks.strftime("%Y"), rotation=45, fontsize=12)
            ax.grid(True)
            ax.set_title(f"Label: {label}, Rating: {rating}", fontsize=14)

            if j == 0:
                ax.set_ylabel("Mean SRC")

    plt.subplots_adjust(top=0.88, bottom=0.1, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
    fig_c.savefig(OUT_DIR / "C_mean_src_by_rating.pdf", bbox_inches="tight")
    fig_c.savefig(OUT_DIR / "C_mean_src_by_rating.png", bbox_inches="tight", dpi=600)
    plt.close(fig_c)

    fig_d, axes_d = plt.subplots(nrows=2, ncols=5, figsize=(16, 6), sharex=True, sharey=True)
    norm = plt.Normalize(vmin=0, vmax=1.5)
    cmap = "viridis"
    for i, label in enumerate(["fake", "real"]):
        for j, rating in enumerate([1, 2, 3, 4, 5]):
            ax = axes_d[i, j]
            subset = daily_stats_rating[
                (daily_stats_rating["Label"] == label) &
                (daily_stats_rating["Rating"] == rating)
            ]
            ax.scatter(
                subset["date"],
                subset["std_SRC"],
                c=subset["cv"],
                cmap=cmap,
                s=(subset["cv"] + 0.1) * 140,
                alpha=0.6,
                edgecolors="face",
                vmin=0, vmax=1.5
            )
            ax.set_xlim(start_date, end_date)
            ax.set_ylim(0, 1)
            ax.set_xticks(year_ticks)
            ax.set_xticklabels(year_ticks.strftime("%Y"), rotation=45, fontsize=12)
            ax.grid(True)
            ax.set_title(f"Label: {label}, Rating: {rating}", fontsize=14)

            if j == 0:
                ax.set_ylabel("Std SRC")

    plt.subplots_adjust(top=0.88, bottom=0.1, left=0.05, right=0.90, hspace=0.4, wspace=0.3)

    cbar_ax = fig_d.add_axes((0.92, 0.1, 0.01, 0.78))
    cv_ticks = np.arange(0, 1.6, 0.3)
    fig_d.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        ticks=cv_ticks,
        label="CV (Coefficient of Variation)"
    )

    fig_d.savefig(OUT_DIR / "D_std_cv_by_rating.pdf", bbox_inches="tight")
    fig_d.savefig(OUT_DIR / "D_std_cv_by_rating.png", bbox_inches="tight", dpi=600)
    plt.close(fig_d)

    rating_counts = df.groupby(["Rating", "Label"]).size().unstack(fill_value=0)
    rating_counts["total"] = rating_counts.sum(axis=1)
    rating_counts["real_prop"] = rating_counts["real"] / rating_counts["total"]
    rating_counts["fake_prop"] = rating_counts["fake"] / rating_counts["total"]

    fig_e, ax = plt.subplots(figsize=(8, 6))
    ratings = rating_counts.index.tolist()
    x = np.arange(len(ratings))
    group_width = 0.8

    real_outer_x = x - group_width / 4
    fake_outer_x = x + group_width / 4
    outer_width = group_width / 2

    ax.bar(
        real_outer_x,
        rating_counts["real_prop"],
        width=outer_width,
        color="none",
        edgecolor="#e41a1c",
        hatch="///",
        label="Real Reviews (%)"
    )
    ax.bar(
        fake_outer_x,
        rating_counts["fake_prop"],
        width=outer_width,
        color="none",
        edgecolor="#377eb8",
        hatch="///",
        label="Fake Reviews (%)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(ratings)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    inner_width = group_width / 3
    ax2.bar(
        real_outer_x,
        rating_counts["real"],
        width=inner_width,
        color="#e41a1c",
        alpha=0.3,
        label="Real Reviews (Count)"
    )
    ax2.bar(
        fake_outer_x,
        rating_counts["fake"],
        width=inner_width,
        color="#377eb8",
        alpha=0.3,
        label="Fake Reviews (Count)"
    )
    ax2.set_ylabel("Count")
    ax2.set_ylim(0, 100000)
    ax2.set_yticks([0, 20000, 40000, 60000, 80000, 100000])

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    fig_e.savefig(OUT_DIR / "E_nested_bars_rating_real_fake.pdf", bbox_inches="tight")
    fig_e.savefig(OUT_DIR / "E_nested_bars_rating_real_fake.png", bbox_inches="tight", dpi=600)
    plt.close(fig_e)

if __name__ == "__main__":
    main()