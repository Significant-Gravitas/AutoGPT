# list all folders in benchmark folder
# for each folder, run the benchmark
import contextlib
import json
import os
import subprocess

from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Iterable, Union

from tabulate import tabulate
from typer import run


def main(
    n_benchmarks: Union[int, None] = None,
):
    path = Path("benchmark")

    folders: Iterable[Path] = path.iterdir()

    if n_benchmarks:
        folders = islice(folders, n_benchmarks)

    benchmarks = []
    results = []
    for bench_folder in folders:
        if os.path.isdir(bench_folder):
            print(f"Running benchmark for {bench_folder}")

            log_path = bench_folder / "log.txt"
            log_file = open(log_path, "w")
            process = subprocess.Popen(
                [
                    "python",
                    "-u",  # Unbuffered output
                    "-m",
                    "gpt_engineer.cli.main",
                    bench_folder,
                    "--steps",
                    "benchmark",
                ],
                stdout=log_file,
                stderr=log_file,
                bufsize=0,
            )
            benchmarks.append(bench_folder)
            results.append((process, log_file))

            print("You can stream the log file by running:")
            print(f"tail -f {log_path}")
            print()

    for bench_folder, (process, file) in zip(benchmarks, results):
        process.wait()
        file.close()

        print("process", bench_folder.name, "finished with code", process.returncode)
        print("Running it. Original benchmark prompt:")
        print()
        with open(bench_folder / "prompt") as f:
            print(f.read())
        print()

        with contextlib.suppress(KeyboardInterrupt):
            subprocess.run(
                [
                    "python",
                    "-m",
                    "gpt_engineer.cli.main",
                    bench_folder,
                    "--steps",
                    "evaluate",
                ],
            )

    generate_report(benchmarks, path)


def generate_report(benchmarks, benchmark_path):
    headers = ["Benchmark", "Ran", "Works", "Perfect", "Notes"]
    rows = []
    for bench_folder in benchmarks:
        memory = bench_folder / ".gpteng" / "memory"
        with open(memory / "review") as f:
            review = json.loads(f.read())
            rows.append(
                [
                    bench_folder.name,
                    to_emoji(review.get("ran", None)),
                    to_emoji(review.get("works", None)),
                    to_emoji(review.get("perfect", None)),
                    review.get("comments", None),
                ]
            )
    table: str = tabulate(rows, headers, tablefmt="pipe")
    print("\nBenchmark report:\n")
    print(table)
    print()
    append_to_results = ask_yes_no("Append report to the results file?")
    if append_to_results:
        results_path = benchmark_path / "RESULTS.md"
        current_date = datetime.now().strftime("%Y-%m-%d")
        insert_markdown_section(results_path, current_date, table, 2)


def to_emoji(value: bool) -> str:
    return "\U00002705" if value else "\U0000274C"


def insert_markdown_section(file_path, section_title, section_text, level):
    with open(file_path, "r") as file:
        lines = file.readlines()

    header_prefix = "#" * level
    new_section = f"{header_prefix} {section_title}\n\n{section_text}\n\n"

    # Find the first section with the specified level
    line_number = -1
    for i, line in enumerate(lines):
        if line.startswith(header_prefix):
            line_number = i
            break

    if line_number != -1:
        lines.insert(line_number, new_section)
    else:
        print(
            f"Markdown file was of unexpected format. No section of level {level} found. "
            "Did not write results."
        )
        return

    # Write the file
    with open(file_path, "w") as file:
        file.writelines(lines)


def ask_yes_no(question: str) -> bool:
    while True:
        response = input(question + " (y/n): ").lower().strip()
        if response == "y":
            return True
        elif response == "n":
            return False
        else:
            print("Please enter either 'y' or 'n'.")


if __name__ == "__main__":
    run(main)
