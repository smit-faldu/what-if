"""
What If — Automated Content Creation Pipeline
CLI entry point.

Usage:
    python main.py                   # brainstorm 10 ideas (default)
    python main.py --count 20        # brainstorm 20 ideas
    python main.py --count 15 --run-id my-custom-id
"""

from __future__ import annotations
import argparse
import uuid
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import DEFAULT_IDEA_COUNT
from pipeline.graph import build_graph

console = Console()

TYPE_STYLE = {
    "hook":   ("🎣", "red",    "HOOK"),
    "body":   ("🎬", "cyan",   "BODY"),
    "closer": ("🤯", "green",  "CLOSER"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a What If video script (7-10 TTS dialogs + Flux image prompts)."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_IDEA_COUNT,
        help=f"Number of ideas to brainstorm (default: {DEFAULT_IDEA_COUNT})",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (default: auto-generated UUID)",
    )
    return parser.parse_args()


def print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold magenta]✨ WHAT IF — Content Creation Pipeline[/bold magenta]\n"
            "[dim]Gemini · LangGraph · Supabase pgvector · all-MiniLM embeddings[/dim]",
            border_style="magenta",
            padding=(1, 4),
        )
    )


def print_dialogs(state: dict) -> None:
    idea = state.get("selected_idea", "")
    reason = state.get("selection_reason", "")
    script = state.get("script", {})
    dialogs = script.get("dialogs", [])
    duration = script.get("total_duration_estimate", "?")

    # ── Idea banner ───────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[bold yellow]💡 {idea}[/bold yellow]\n"
            + (f"[dim italic]{reason}[/dim italic]" if reason else ""),
            title="[bold]CHOSEN IDEA[/bold]",
            border_style="yellow",
            padding=(1, 2),
        )
    )

    # ── Dialog table overview ─────────────────────────────────────────────────
    console.print(f"\n[dim]📋 {len(dialogs)} dialog lines  |  ⏱ ~{duration} seconds total[/dim]\n")

    # ── Each dialog line ──────────────────────────────────────────────────────
    for d in dialogs:
        emoji, color, label = TYPE_STYLE.get(d["dialog_type"], ("📌", "white", d["dialog_type"].upper()))

        console.print(
            Panel(
                f"[bold white]{d['text']}[/bold white]\n\n"
                f"[dim]Flux ↓[/dim]\n[{color}]{d['flux_prompt']}[/{color}]",
                title=f"[bold {color}]{emoji} Line {d['line_number']}  [{label}][/bold {color}]",
                border_style=color,
                padding=(1, 2),
            )
        )

    # ── TTS preview ───────────────────────────────────────────────────────────
    console.print()
    tts_table = Table(
        title="📢 TTS Lines (copy → paste into your TTS tool)",
        box=box.SIMPLE_HEAVY,
        border_style="dim",
        show_lines=True,
    )
    tts_table.add_column("#", style="dim", width=3)
    tts_table.add_column("Type", width=8)
    tts_table.add_column("Dialog Text", style="bold white")

    for d in dialogs:
        emoji, color, label = TYPE_STYLE.get(d["dialog_type"], ("📌", "white", "?"))
        tts_table.add_row(
            str(d["line_number"]),
            f"[{color}]{emoji} {label}[/{color}]",
            d["text"],
        )
    console.print(tts_table)


def print_summary(state: dict, run_id: str) -> None:
    output_path = state.get("output_path", "")
    console.print()
    console.print(
        Panel(
            f"[bold green]✅ Pipeline complete![/bold green]\n\n"
            f"Run ID    : [bold]{run_id}[/bold]\n"
            f"Output    : [bold cyan]{output_path}[/bold cyan]\n\n"
            f"[dim]Files saved:\n"
            f"  • dialogs.md       ← script + image prompts per line\n"
            f"  • tts_lines.txt    ← clean text for TTS (one line per clip)\n"
            f"  • data.json        ← full structured output[/dim]",
            border_style="green",
            padding=(1, 2),
        )
    )


def main() -> None:
    args = parse_args()
    run_id = args.run_id or str(uuid.uuid4())

    print_banner()
    console.print(f"\n[dim]Run ID   : {run_id}[/dim]")
    console.print(f"[dim]Candidates: {args.count} ideas to brainstorm[/dim]\n")

    app = build_graph()

    initial_state: dict = {
        "run_id": run_id,
        "idea_count": args.count,
        "candidates": [],
        "filtered_candidates": [],
        "selected_idea": "",
        "selection_reason": "",
        "script": None,
        "output_path": None,
    }

    config = {"configurable": {"thread_id": run_id}}

    with console.status("[bold magenta]Running pipeline...[/bold magenta]", spinner="dots"):
        try:
            final_state = app.invoke(initial_state, config=config)
        except Exception as e:
            console.print(f"\n[bold red]❌ Pipeline failed:[/bold red] {e}")
            raise

    print_dialogs(final_state)
    print_summary(final_state, run_id)


if __name__ == "__main__":
    main()
