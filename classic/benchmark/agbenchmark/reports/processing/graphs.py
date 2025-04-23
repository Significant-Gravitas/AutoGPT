from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_combined_radar_chart(
    categories: dict[str, Any], save_path: str | Path
) -> None:
    categories = {k: v for k, v in categories.items() if v}
    if not all(categories.values()):
        raise Exception("No data to plot")
    labels = np.array(
        list(next(iter(categories.values())).keys())
    )  # We use the first category to get the keys
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[
        :1
    ]  # Add the first angle to the end of the list to ensure the polygon is closed

    # Create radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)  # type: ignore
    ax.set_theta_direction(-1)  # type: ignore
    ax.spines["polar"].set_visible(False)  # Remove border

    cmap = plt.cm.get_cmap("nipy_spectral", len(categories))  # type: ignore

    colors = [cmap(i) for i in range(len(categories))]

    for i, (cat_name, cat_values) in enumerate(
        categories.items()
    ):  # Iterating through each category (series)
        values = np.array(list(cat_values.values()))
        values = np.concatenate((values, values[:1]))  # Ensure the polygon is closed

        ax.fill(angles, values, color=colors[i], alpha=0.25)  # Draw the filled polygon
        ax.plot(angles, values, color=colors[i], linewidth=2)  # Draw polygon
        ax.plot(
            angles,
            values,
            "o",
            color="white",
            markersize=7,
            markeredgecolor=colors[i],
            markeredgewidth=2,
        )  # Draw points

        # Draw legend
        ax.legend(
            handles=[
                mpatches.Patch(color=color, label=cat_name, alpha=0.25)
                for cat_name, color in zip(categories.keys(), colors)
            ],
            loc="upper left",
            bbox_to_anchor=(0.7, 1.3),
        )

        # Adjust layout to make room for the legend
        plt.tight_layout()

    lines, labels = plt.thetagrids(
        np.degrees(angles[:-1]), (list(next(iter(categories.values())).keys()))
    )  # We use the first category to get the keys

    highest_score = 7

    # Set y-axis limit to 7
    ax.set_ylim(top=highest_score)

    # Move labels away from the plot
    for label in labels:
        label.set_position(
            (label.get_position()[0], label.get_position()[1] + -0.05)
        )  # adjust 0.1 as needed

    # Move radial labels away from the plot
    ax.set_rlabel_position(180)  # type: ignore

    ax.set_yticks([])  # Remove default yticks

    # Manually create gridlines
    for y in np.arange(0, highest_score + 1, 1):
        if y != highest_score:
            ax.plot(
                angles, [y] * len(angles), color="gray", linewidth=0.5, linestyle=":"
            )
        # Add labels for manually created gridlines
        ax.text(
            angles[0],
            y + 0.2,
            str(int(y)),
            color="black",
            size=9,
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.savefig(save_path, dpi=300)  # Save the figure as a PNG file
    plt.close()  # Close the figure to free up memory


def save_single_radar_chart(
    category_dict: dict[str, int], save_path: str | Path
) -> None:
    labels = np.array(list(category_dict.keys()))
    values = np.array(list(category_dict.values()))

    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    angles += angles[:1]
    values = np.concatenate((values, values[:1]))

    colors = ["#1f77b4"]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)  # type: ignore
    ax.set_theta_direction(-1)  # type: ignore

    ax.spines["polar"].set_visible(False)

    lines, labels = plt.thetagrids(
        np.degrees(angles[:-1]), (list(category_dict.keys()))
    )

    highest_score = 7

    # Set y-axis limit to 7
    ax.set_ylim(top=highest_score)

    for label in labels:
        label.set_position((label.get_position()[0], label.get_position()[1] + -0.05))

    ax.fill(angles, values, color=colors[0], alpha=0.25)
    ax.plot(angles, values, color=colors[0], linewidth=2)

    for i, (angle, value) in enumerate(zip(angles, values)):
        ha = "left"
        if angle in {0, np.pi}:
            ha = "center"
        elif np.pi < angle < 2 * np.pi:
            ha = "right"
        ax.text(
            angle,
            value - 0.5,
            f"{value}",
            size=10,
            horizontalalignment=ha,
            verticalalignment="center",
            color="black",
        )

    ax.set_yticklabels([])

    ax.set_yticks([])

    if values.size == 0:
        return

    for y in np.arange(0, highest_score, 1):
        ax.plot(angles, [y] * len(angles), color="gray", linewidth=0.5, linestyle=":")

    for angle, value in zip(angles, values):
        ax.plot(
            angle,
            value,
            "o",
            color="white",
            markersize=7,
            markeredgecolor=colors[0],
            markeredgewidth=2,
        )

    plt.savefig(save_path, dpi=300)  # Save the figure as a PNG file
    plt.close()  # Close the figure to free up memory


def save_combined_bar_chart(categories: dict[str, Any], save_path: str | Path) -> None:
    if not all(categories.values()):
        raise Exception("No data to plot")

    # Convert dictionary to DataFrame
    df = pd.DataFrame(categories)

    # Create a grouped bar chart
    df.plot(kind="bar", figsize=(10, 7))

    plt.title("Performance by Category for Each Agent")
    plt.xlabel("Category")
    plt.ylabel("Performance")

    plt.savefig(save_path, dpi=300)  # Save the figure as a PNG file
    plt.close()  # Close the figure to free up memory
