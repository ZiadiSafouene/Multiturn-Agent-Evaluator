"""
core.cli
~~~~~~~~

Command-line interface for running conversational agent evaluations.

Published to PyPI with executable console script entrypoint ``agenteval``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.__version__ import __version__
from core.orchestrator import AgentEvaluator, EvaluatorConfig
from core.utils.models import Language, Persona

app = typer.Typer(
    name="agenteval",
    help="Multi-turn conversational AI agent evaluator with multilingual & adversarial testing.",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    if value:
        rprint(f"[bold cyan]AgentEval[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    pass


@app.command(name="evaluate", help="Evaluate an agent across multi-turn scenarios.")
def evaluate_cmd(
    scenarios: str = typer.Option(
        "test_scenarios.json",
        "--scenarios",
        "-s",
        help="Path to scenarios JSON file OR a natural language description to synthesize scenarios from.",
    ),
    agent_key: Optional[str] = typer.Option(
        None,
        "--agent-key",
        "-k",
        envvar="AGENT_API_KEY",
        help="OpenRouter API key for agent under test (optional; uses free model if omitted).",
    ),
    agent_model: str = typer.Option(
        "mistralai/mistral-7b-instruct:free",
        "--agent-model",
        "-m",
        envvar="AGENT_MODEL",
        help="OpenRouter model ID for the agent under test.",
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system-prompt",
        "-p",
        help="System prompt defining the agent under test persona and guidelines.",
    ),
    tester_key: Optional[str] = typer.Option(
        None,
        "--tester-key",
        envvar="TESTER_API_KEY",
        help="OpenRouter API key for the tester & judge model (optional; uses free model if omitted).",
    ),
    tester_model: str = typer.Option(
        "mistralai/mistral-7b-instruct:free",
        "--tester-model",
        envvar="TESTER_MODEL",
        help="OpenRouter model ID for user simulation and judging.",
    ),
    languages: List[str] = typer.Option(
        ["english", "arabic", "french"],
        "--language",
        "-l",
        help="Languages to test: english, arabic, french.",
    ),
    cases_per_scenario: int = typer.Option(
        2,
        "--cases",
        "-n",
        help="Number of test cases to generate per scenario.",
    ),
    judge: bool = typer.Option(
        False,
        "--judge/--no-judge",
        help="Enable qualitative LLM-as-a-judge scoring in addition to deterministic metrics.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory to write detailed JSON evaluation reports.",
    ),
):
    """Run full evaluation suite on the agent."""
    banner = Panel.fit(
        f"[bold cyan]AgentEval v{__version__}[/bold cyan]\n"
        f"[dim]Multi-turn conversational agent evaluation framework[/dim]\n\n"
        f"[bold]Agent Model :[/bold] [yellow]{agent_model}[/yellow] "
        f"({'Custom API Key' if agent_key else 'Default Free Model'})\n"
        f"[bold]Tester Model:[/bold] [yellow]{tester_model}[/yellow] "
        f"({'Custom API Key' if tester_key else 'Default Free Model'})\n"
        f"[bold]Languages   :[/bold] {', '.join(languages)}\n"
        f"[bold]Scenarios   :[/bold] {scenarios}",
        title="[bold green]Evaluation Configuration[/bold green]",
        border_style="cyan",
    )
    console.print(banner)

    # Map language strings to enum
    target_languages: List[Language] = []
    for lang_str in languages:
        try:
            target_languages.append(Language(lang_str.lower()))
        except ValueError:
            rprint(f"[red]Warning:[/red] Unknown language '{lang_str}'. Ignored.")

    if not target_languages:
        target_languages = [Language.ENGLISH, Language.ARABIC, Language.FRENCH]

    config = EvaluatorConfig(
        agent_api_key=agent_key,
        agent_model=agent_model,
        agent_system_prompt=system_prompt,
        tester_api_key=tester_key,
        tester_model=tester_model,
        languages=target_languages,
        test_cases_per_scenario=cases_per_scenario,
        use_llm_judge=judge,
        output_dir=output,
    )

    evaluator = AgentEvaluator(config)

    with console.status("[bold green]Executing evaluation pipeline...", spinner="dots"):
        results = evaluator.evaluate(scenarios)

    # Print summary table
    table = Table(title="Evaluation Suite Results", border_style="green")
    table.add_column("Test ID", style="cyan")
    table.add_column("Language", style="magenta")
    table.add_column("Persona", style="blue")
    table.add_column("Mutation", style="yellow")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Status", style="bold")

    for report in results.reports:
        status_color = "green" if report.status.value == "pass" else ("yellow" if report.status.value == "partial" else "red")
        table.add_row(
            report.test_case_id,
            report.language.value.capitalize() if report.language else "N/A",
            report.persona.value.capitalize() if report.persona else "N/A",
            report.mutation.type.value if report.mutation else "none",
            f"{report.final_score:.1f}",
            f"[{status_color}]{report.status.value.upper()}[/{status_color}]",
        )

    console.print(table)

    summary_panel = Panel.fit(
        f"[bold]Total Tests :[/bold] {results.total_tests}\n"
        f"[bold green]Passed      :[/bold green] {results.passed_tests}\n"
        f"[bold yellow]Partial     :[/bold yellow] {results.partial_tests}\n"
        f"[bold red]Failed      :[/bold red] {results.failed_tests}\n"
        f"[bold cyan]Average Score:[/bold cyan] {results.average_score:.1f}/100\n\n"
        f"[dim]Scores by Language:[/dim] {results.scores_by_language}\n"
        f"[dim]Scores by Persona :[/dim] {results.scores_by_persona}",
        title="[bold]Summary[/bold]",
        border_style="green" if results.average_score >= 70 else "red",
    )
    console.print(summary_panel)


if __name__ == "__main__":
    app()

