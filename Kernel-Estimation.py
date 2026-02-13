import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

mpl.rcParams["savefig.format"] = "pdf"
OUT_DIR = Path("figures_pdf_png")
OUT_DIR.mkdir(parents=True, exist_ok=True)

inset_position_x = 0.3
inset_position_y = 4
inset_width = 2
inset_height = 1.5

inset_x_range = (0, 0.2)
inset_y_range = (0, 0.2)

def main():
    data_file = "data.xlsx"
    df = pd.read_excel(data_file)

    required_cols = {"Date", "Reviewer_id", "Rating", "Sentiment", "Label_user"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    if pd.api.types.is_numeric_dtype(df["Label_user"]):
        df["Label_user"] = df["Label_user"].map({0: "real", 1: "fake"})

    df["rating_norm"] = (df["Rating"] - 1) / 4.0
    df["SRC"] = 1 - np.abs(df["Sentiment"] - df["rating_norm"])
    df["SRC"] = df["SRC"].clip(0, 1)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    sns.kdeplot(
        data=df[df["Label_user"] == "real"],
        x="SRC",
        clip=(0, 1),
        fill=True,
        color="#e41a1c",
        label="Real Users",
        alpha=0.5,
        common_norm=False,
        ax=ax
    )
    sns.kdeplot(
        data=df[df["Label_user"] == "fake"],
        x="SRC",
        clip=(0, 1),
        fill=True,
        color="#377eb8",
        label="Fake Users",
        alpha=0.5,
        common_norm=False,
        ax=ax
    )

    center_x = inset_position_x
    center_y = inset_position_y

    axins = inset_axes(
        ax,
        width=inset_width,
        height=inset_height,
        loc="center",
        bbox_to_anchor=(center_x - inset_width / 2, center_y - inset_height / 2, inset_width, inset_height),
        bbox_transform=ax.transData
    )

    sns.kdeplot(
        data=df[df["Label_user"] == "real"],
        x="SRC",
        clip=inset_x_range,
        fill=True,
        color="#e41a1c",
        alpha=0.5,
        common_norm=False,
        ax=axins
    )
    sns.kdeplot(
        data=df[df["Label_user"] == "fake"],
        x="SRC",
        clip=inset_x_range,
        fill=True,
        color="#377eb8",
        alpha=0.5,
        common_norm=False,
        ax=axins
    )

    axins.set_xlim(inset_x_range)
    axins.set_ylim(inset_y_range)
    axins.set_yticks(np.linspace(inset_y_range[0], inset_y_range[1], 5))
    axins.set_xticks(np.linspace(inset_x_range[0], inset_x_range[1], 3))
    axins.set_xlabel("")
    axins.set_ylabel("")

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black")

    ax.set_xlabel("SRC")
    ax.set_ylabel("Density")
    ax.legend(title="User Type")
    ax.grid(True)
    ax.set_xlim(0, 1)

    outfile_pdf = OUT_DIR / "kde_src_with_inset.pdf"
    outfile_png = OUT_DIR / "kde_src_with_inset.png"

    fig.savefig(outfile_pdf, bbox_inches="tight")
    fig.savefig(outfile_png, bbox_inches="tight", dpi=600)

    plt.close(fig)
    print(f"Saved: {outfile_pdf} and {outfile_png}")

if __name__ == "__main__":
    main()
