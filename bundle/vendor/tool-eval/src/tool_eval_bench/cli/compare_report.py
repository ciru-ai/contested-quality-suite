"""CLI support for generating browser HTML comparison reports."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console


def detect_report_kind(path: str) -> str:
    """Return ``summary`` or ``tool-eval`` based on the report heading."""
    first_lines = Path(path).read_text(encoding="utf-8").splitlines()[:10]
    for line in first_lines:
        if line.startswith("# Cross-Trial Summary"):
            return "summary"
        if line.startswith("# Tool-Call Benchmark"):
            return "tool-eval"
    raise ValueError(
        f"Could not detect report type for {path}. Expected '# Cross-Trial Summary' "
        "or '# Tool-Call Benchmark'."
    )


def generate_compare_report(
    report_a: str,
    report_b: str,
    output: str,
    *,
    kind: str = "auto",
) -> str:
    """Generate an HTML comparison report and return the selected report kind."""
    for report in (report_a, report_b):
        if not Path(report).exists():
            raise FileNotFoundError(f"file not found: {report}")

    selected_kind = kind
    if kind == "auto":
        kind_a = detect_report_kind(report_a)
        kind_b = detect_report_kind(report_b)
        if kind_a != kind_b:
            raise ValueError(
                f"Report types do not match: {report_a} is {kind_a}, {report_b} is {kind_b}."
            )
        selected_kind = kind_a

    if selected_kind == "summary":
        from tool_eval_bench.compare_reports.summary import generate_html, parse_summary

        generate_html(parse_summary(report_a), parse_summary(report_b), output)
    elif selected_kind == "tool-eval":
        from tool_eval_bench.compare_reports.tool_eval import generate_html, parse_md

        generate_html(parse_md(report_a), parse_md(report_b), output)
    else:
        raise ValueError(f"unknown report kind: {kind}")

    return selected_kind


def run_compare_report_command(args, console: Console) -> None:
    """Run the ``compare-report`` CLI subcommand."""
    try:
        kind = generate_compare_report(
            args.report_a,
            args.report_b,
            args.output,
            kind=args.kind,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from exc

    console.print(f"[bold green]✓[/] Generated {kind} comparison report: {args.output}")
