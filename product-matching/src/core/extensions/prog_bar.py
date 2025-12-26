from typing import List, Dict, Union, Any
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    ProgressColumn,
    SpinnerColumn,
)
from rich.text import Text


class SpeedColumn(ProgressColumn):
    def render(self, task: "Task") -> Text:
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?/s", style="progress.data.speed")
        return Text(f" {speed:.2f}/s ", style="progress.data.speed")


class MetricsColumn(ProgressColumn):
    def __init__(self, ignore_prefix: Union[str, List[str]] = []):
        super().__init__()
        if isinstance(ignore_prefix, str):
            ignore_prefix = [ignore_prefix]
        self.ignore_prefix = ["_"] + ignore_prefix

    def render(self, task: "Task") -> Text:
        fields = task.fields
        render_text = ""
        if len(fields):
            for key, val in fields.items():
                if not any([key.startswith(pre) for pre in self.ignore_prefix]):
                    if isinstance(val, float):
                        val = f"{val:.3f}"
                    render_text += f"| {key} : {val} "
        return Text(render_text, style="blue")


class ProgressBar:
    def __init__(self, total, description, ignore_prefix=""):
        self.progress = Progress(
            SpinnerColumn("dots12", speed=2.0),
            TextColumn(f"{description} | "),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn(f" | ETA: "),
            TimeElapsedColumn(),
            TextColumn(f" /"),
            TimeRemainingColumn(),
            TextColumn(f"| Batch:"),
            MofNCompleteColumn(" / "),
            SpeedColumn(),
            MetricsColumn(ignore_prefix=ignore_prefix),
        )
        self.task = self.progress.add_task("task", total=total)

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()
        print()

    def advance(self):
        self.progress.update(self.task, advance=1)

    def log(self, metrics: Dict[str, Any]):
        self.progress.update(self.task, **metrics)
