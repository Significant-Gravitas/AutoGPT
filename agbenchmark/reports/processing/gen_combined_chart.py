import os
from pathlib import Path

from agbenchmark.reports.processing.graphs import save_combined_radar_chart
from agbenchmark.reports.processing.process_report import (
    all_agent_categories,
    get_reports_data,
)
from agbenchmark.start_benchmark import REPORTS_PATH


def generate_combined_chart() -> None:
    reports_data = get_reports_data(REPORTS_PATH)

    categories = all_agent_categories(reports_data)

    png_count = len([f for f in os.listdir(REPORTS_PATH) if f.endswith(".png")])

    save_combined_radar_chart(
        categories, Path(REPORTS_PATH) / f"run{png_count + 1}_radar_chart.png"
    )


if __name__ == "__main__":
    generate_combined_chart()
