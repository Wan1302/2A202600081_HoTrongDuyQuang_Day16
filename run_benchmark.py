from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(dataset: str = "data/hotpot_mini.json", out_dir: str = "outputs/sample_run", reflexion_attempts: int = 3, model: str = "gpt-4o-mini", use_mock: bool = False) -> None:
    examples = load_dataset(dataset)
    react = ReActAgent(model=model, use_mock=use_mock)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, model=model, use_mock=use_mock)
    react_records = [react.run(example) for example in examples]
    reflexion_records = [reflexion.run(example) for example in examples]
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report_mode = "mock" if (use_mock or getattr(react, "runtime_mode", "mock") == "mock" or getattr(reflexion, "runtime_mode", "mock") == "mock") else "openai"
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=report_mode, model=model)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
