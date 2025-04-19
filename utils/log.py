from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

CONSOLE = Console(highlight=False)


class TaskSpeedColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def render(self, task: Task, unit="it") -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:>.3f} {unit}/s", style="progress.data.speed")


def progress_columns():
    """Custom rich.Progress columns, showing speed and elapsed time."""
    return (
        TextColumn("{task.description}"),
        BarColumn(),
        TaskSpeedColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(
            # add parentheses around percentage
            text_format="([progress.percentage]{task.percentage:>3.0f}%)",
            show_speed=True,
        ),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )
