#!/usr/bin/env python3

from pathlib import Path

import click

from agbenchmark.reports.processing.report_types import Report


@click.command()
@click.argument(
    "report_json_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def print_markdown_report(report_json_file: Path):
    """
    Generates a Markdown report from a given report.json file.

    :param report_json_file: Path to the report.json file.
    :return: A string containing the Markdown formatted report.
    """
    report = Report.model_validate_json(report_json_file.read_text())

    # Header and metadata
    click.echo("# Benchmark Report")
    click.echo(f"- ⌛ **Run time:** `{report.metrics.run_time}`")
    click.echo(
        f"  - **Started at:** `{report.benchmark_start_time[:16].replace('T', '` `')}`"
    )
    if report.completion_time:
        click.echo(
            f"  - **Completed at:** `{report.completion_time[:16].replace('T', '` `')}`"
        )
    if report.metrics.total_cost:
        click.echo(f"- 💸 **Total cost:** `${round(report.metrics.total_cost, 2)}`")
    click.echo(
        f"- 🏅 **Highest achieved difficulty:** `{report.metrics.highest_difficulty}`"
    )
    click.echo(f"- ⚙️ **Command:** `{report.command}`")

    click.echo()  # spacing

    # Aggregate information
    successful, failed, unreliable = [], [], []
    for test in report.tests.values():
        test.metrics.success_percentage = (
            rsp
            if (rsp := test.metrics.success_percentage) is not None
            else sum(float(r.success or 0) for r in test.results)
            * 100
            / len(test.results)
        )
        if test.metrics.success_percentage == 100.0:
            successful.append(test)
        elif test.metrics.success_percentage == 0.0:
            failed.append(test)
        else:
            unreliable.append(test)

    # Summary
    click.echo("## Summary")
    click.echo(f"- **`{len(successful)}` passed** {'✅'*len(successful)}")
    click.echo(f"- **`{len(failed)}` failed** {'❌'*len(failed)}")
    click.echo(f"- **`{len(unreliable)}` unreliable** {'⚠️'*len(unreliable)}")

    click.echo()  # spacing

    # Test results
    click.echo("## Challenges")
    for test_name, test in report.tests.items():
        click.echo()  # spacing

        result_indicator = (
            "✅"
            if test.metrics.success_percentage == 100.0
            else "⚠️"
            if test.metrics.success_percentage > 0
            else "❌"
        )
        click.echo(
            f"### {test_name} {result_indicator if test.metrics.attempted else '❔'}"
        )
        click.echo(f"{test.description}")

        click.echo()  # spacing

        click.echo(f"- **Attempted:** {'Yes 👍' if test.metrics.attempted else 'No 👎'}")
        click.echo(
            f"- **Success rate:** {round(test.metrics.success_percentage)}% "
            f"({len([r for r in test.results if r.success])}/{len(test.results)})"
        )
        click.echo(f"- **Difficulty:** `{test.difficulty}`")
        click.echo(f"- **Categories:** `{'`, `'.join(test.category)}`")
        click.echo(
            f"<details>\n<summary><strong>Task</strong> (click to expand)</summary>\n\n"
            f"{indent('> ', test.task)}\n\n"
            f"Reference answer:\n{indent('> ', test.answer)}\n"
            "</details>"
        )

        click.echo()  # spacing

        click.echo("\n#### Attempts")
        for i, attempt in enumerate(test.results, 1):
            click.echo(
                f"\n{i}. **{'✅ Passed' if attempt.success else '❌ Failed'}** "
                f"in **{attempt.run_time}** "
                f"and **{quantify('step', attempt.n_steps)}**\n"
            )
            if attempt.cost is not None:
                click.echo(f"   - **Cost:** `${round(attempt.cost, 3)}`")
            if attempt.fail_reason:
                click.echo(
                    "   - **Failure reason:**\n"
                    + indent("      > ", attempt.fail_reason)
                    + "\n"
                )
            if attempt.steps:
                click.echo(
                    indent(
                        3 * " ",
                        "<details>\n<summary><strong>Steps</strong></summary>\n",
                    )
                )
                for j, step in enumerate(attempt.steps, 1):
                    click.echo()
                    click.echo(
                        indent(3 * " ", f"{j}. {indent(3*' ', step.output, False)}")
                    )
                click.echo("\n</details>")


def indent(indent: str, text: str, prefix_indent: bool = True) -> str:
    return (indent if prefix_indent else "") + text.replace("\n", "\n" + indent)


def quantify(noun: str, count: int, plural_suffix: str = "s") -> str:
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {noun}{plural_suffix}"


if __name__ == "__main__":
    print_markdown_report()
