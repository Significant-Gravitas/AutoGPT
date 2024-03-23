import json
import os
from pathlib import Path

from agbenchmark.reports.processing.graphs import (
    save_combined_bar_chart,
    save_combined_radar_chart,
)
from agbenchmark.reports.processing.process_report import (
    all_agent_categories,
    get_reports_data,
)


def generate_combined_chart() -> None:
    all_agents_path = Path(__file__).parent.parent.parent.parent / "reports"

    combined_charts_folder = all_agents_path / "combined_charts"

    reports_data = get_reports_data(str(all_agents_path))

    categories = all_agent_categories(reports_data)

    # Count the number of directories in this directory
    num_dirs = len([f for f in combined_charts_folder.iterdir() if f.is_dir()])

    run_charts_folder = combined_charts_folder / f"run{num_dirs + 1}"

    if not os.path.exists(run_charts_folder):
        os.makedirs(run_charts_folder)

    info_data = {
        report_name: data.benchmark_start_time
        for report_name, data in reports_data.items()
        if report_name in categories
    }
    with open(Path(run_charts_folder) / "run_info.json", "w") as f:
        json.dump(info_data, f)

    save_combined_radar_chart(categories, Path(run_charts_folder) / "radar_chart.png")
    save_combined_bar_chart(categories, Path(run_charts_folder) / "bar_chart.png")


if __name__ == "__main__":
    generate_combined_chart()
