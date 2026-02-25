"""
ProteinScore Command Line Interface

CLI for protein developability scoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from proteinscore import ProteinScore, __version__
from proteinscore.models import ScoringWeights

app = typer.Typer(
    name="proteinscore",
    help="Unified Protein Developability Scorer",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"ProteinScore version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """ProteinScore: Unified Protein Developability Scorer"""
    pass


@app.command()
def score(
    sequence: Annotated[
        str,
        typer.Argument(
            help="Protein sequence (or path to FASTA file)",
        ),
    ],
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Protein name"),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", "-k", envvar="RUNLAB_API_KEY", help="RunLab API key"),
    ] = None,
    local_only: Annotated[
        bool,
        typer.Option("--local", "-l", help="Use local-only mode (no API calls)"),
    ] = False,
    weights: Annotated[
        Optional[str],
        typer.Option("--weights", "-w", help="Scoring weights (e.g., 'enzyme', 'antibody', 'vaccine')"),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Output file (JSON)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output"),
    ] = False,
) -> None:
    """
    Score a protein sequence for developability.

    Examples:
        proteinscore score MKTAYIAKQRQISFVK...
        proteinscore score sequence.fasta --name "My Protein"
        proteinscore score MKTAY... --weights enzyme --output result.json
    """
    # Handle FASTA file input
    if Path(sequence).exists():
        seq_data = _read_fasta(Path(sequence))
        if not seq_data:
            console.print("[red]Error: Could not read sequence from file[/red]")
            raise typer.Exit(1)
        sequence = seq_data["sequence"]
        if not name:
            name = seq_data.get("name")

    # Parse weights
    scoring_weights = None
    if weights:
        if weights == "enzyme":
            scoring_weights = ScoringWeights.for_enzyme()
        elif weights == "antibody":
            scoring_weights = ScoringWeights.for_antibody()
        elif weights == "vaccine":
            scoring_weights = ScoringWeights.for_vaccine()
        else:
            console.print(f"[yellow]Unknown weights preset: {weights}. Using default.[/yellow]")

    # Initialize scorer
    with console.status("Scoring protein...", spinner="dots"):
        try:
            scorer = ProteinScore(
                api_key=api_key,
                weights=scoring_weights,
                local_only=local_only,
            )
            result = scorer.score(sequence, name=name)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e

    # Display results
    _display_result(result, verbose=verbose)

    # Save output
    if output:
        result.to_json(output)
        console.print(f"\n[green]Result saved to {output}[/green]")


@app.command()
def batch(
    input_file: Annotated[
        Path,
        typer.Argument(help="Input file (FASTA or CSV with 'sequence' column)"),
    ],
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", "-k", envvar="RUNLAB_API_KEY", help="RunLab API key"),
    ] = None,
    local_only: Annotated[
        bool,
        typer.Option("--local", "-l", help="Use local-only mode (no API calls)"),
    ] = False,
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Output file (CSV)"),
    ] = None,
    parallel: Annotated[
        bool,
        typer.Option("--parallel/--sequential", help="Enable parallel processing"),
    ] = True,
) -> None:
    """
    Score multiple proteins from a file.

    Examples:
        proteinscore batch sequences.fasta --output results.csv
        proteinscore batch variants.csv --parallel
    """
    # Read input file
    if input_file.suffix.lower() in (".fasta", ".fa", ".faa"):
        sequences = _read_fasta_multi(input_file)
    elif input_file.suffix.lower() == ".csv":
        sequences = _read_csv_sequences(input_file)
    else:
        console.print(f"[red]Unsupported file format: {input_file.suffix}[/red]")
        raise typer.Exit(1)

    if not sequences:
        console.print("[red]No sequences found in input file[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(sequences)} sequences to score")

    # Initialize scorer
    scorer = ProteinScore(
        api_key=api_key,
        local_only=local_only,
    )

    # Score all sequences
    with console.status(f"Scoring {len(sequences)} proteins...", spinner="dots"):
        results = scorer.score_batch(
            sequences=[s["sequence"] for s in sequences],
            names=[s.get("name") for s in sequences],
            parallel=parallel,
        )

    # Display comparison table
    comparison = scorer.compare(results, output_format="table")
    console.print("\n" + comparison)

    # Save output
    if output:
        import csv

        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Name", "Length", "Total", "Stability", "Solubility",
                "Aggregation", "Immunogenicity", "Risk",
            ])
            for r in results:
                writer.writerow([
                    r.name,
                    r.sequence_length,
                    r.total_score,
                    r.stability.score,
                    r.solubility.score,
                    r.aggregation.score,
                    r.immunogenicity.score,
                    r.risk_level.value,
                ])

        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def info() -> None:
    """Show ProteinScore information and configuration."""
    from proteinscore.config import get_default_config

    config = get_default_config()

    console.print("\n[bold]ProteinScore Information[/bold]\n")
    console.print(f"Version: {__version__}")
    console.print(f"API Base URL: {config.api_base_url}")
    console.print(f"API Key: {'configured' if config.is_authenticated else 'not configured'}")
    console.print(f"Tier: {config.tier}")
    console.print(f"Cache: {'enabled' if config.cache_enabled else 'disabled'}")
    console.print(f"Cache Directory: {config.cache_dir}")


def _display_result(result, verbose: bool = False) -> None:
    """Display scoring result."""
    # Main score
    risk_color = {
        "low": "green",
        "medium": "yellow",
        "high": "red",
    }[result.risk_level.value]

    console.print(f"\n[bold]ProteinScore Result[/bold]")
    if result.name:
        console.print(f"Name: {result.name}")
    console.print(f"Length: {result.sequence_length} residues")
    console.print(f"Total Score: [bold]{result.total_score:.1f}[/bold]/100")
    console.print(f"Risk Level: [{risk_color}][bold]{result.risk_level.value.upper()}[/bold][/{risk_color}]")

    # Component scores table
    table = Table(title="\nComponent Scores")
    table.add_column("Component", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Interpretation")

    table.add_row(
        "Stability",
        f"{result.stability.score:.1f}",
        result.stability.interpretation,
    )
    table.add_row(
        "Solubility",
        f"{result.solubility.score:.1f}",
        result.solubility.interpretation,
    )
    table.add_row(
        "Aggregation",
        f"{result.aggregation.score:.1f}",
        result.aggregation.interpretation,
    )
    table.add_row(
        "Immunogenicity",
        f"{result.immunogenicity.score:.1f}",
        result.immunogenicity.interpretation,
    )

    console.print(table)

    # Recommendations
    if result.recommendations:
        console.print("\n[bold]Recommendations[/bold]")
        for rec in result.recommendations[:5]:
            severity_color = {
                "critical": "red",
                "warning": "yellow",
                "info": "blue",
            }.get(rec.severity, "white")
            console.print(f"  [{severity_color}][{rec.severity.upper()}][/{severity_color}] {rec.message}")

    if verbose:
        # Additional details
        console.print("\n[bold]Details[/bold]")
        console.print(f"Stability method: {result.stability.method}")
        console.print(f"Solubility method: {result.solubility.method}")
        console.print(f"Aggregation method: {result.aggregation.method}")
        console.print(f"Immunogenicity method: {result.immunogenicity.method}")

        if result.stability.unstable_regions:
            console.print(f"\nUnstable regions: {len(result.stability.unstable_regions)}")
        if result.aggregation.aggregation_prone_regions:
            console.print(f"APRs: {len(result.aggregation.aggregation_prone_regions)}")
        if result.immunogenicity.epitope_count:
            console.print(f"Epitopes: {result.immunogenicity.epitope_count}")


def _read_fasta(path: Path) -> dict[str, str] | None:
    """Read single sequence from FASTA file."""
    with open(path) as f:
        lines = f.readlines()

    name = None
    sequence_parts = []

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
        elif line:
            sequence_parts.append(line)

    if sequence_parts:
        return {"name": name, "sequence": "".join(sequence_parts)}
    return None


def _read_fasta_multi(path: Path) -> list[dict[str, str]]:
    """Read multiple sequences from FASTA file."""
    sequences = []

    with open(path) as f:
        lines = f.readlines()

    current_name = None
    current_seq: list[str] = []

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if current_seq:
                sequences.append({
                    "name": current_name,
                    "sequence": "".join(current_seq),
                })
            current_name = line[1:].split()[0]
            current_seq = []
        elif line:
            current_seq.append(line)

    if current_seq:
        sequences.append({
            "name": current_name,
            "sequence": "".join(current_seq),
        })

    return sequences


def _read_csv_sequences(path: Path) -> list[dict[str, str]]:
    """Read sequences from CSV file."""
    import csv

    sequences = []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "sequence" in row:
                sequences.append({
                    "name": row.get("name") or row.get("id"),
                    "sequence": row["sequence"],
                })

    return sequences


if __name__ == "__main__":
    app()
