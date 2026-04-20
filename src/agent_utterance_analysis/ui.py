from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable

THEME_STYLES = {
    "dark": {
        "accent": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "muted": "bright_black",
        "title": "bold cyan",
    },
    "light": {
        "accent": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "muted": "grey50",
        "title": "bold blue",
    },
    "mono": {
        "accent": "white",
        "success": "white",
        "warning": "white",
        "error": "white",
        "muted": "white",
        "title": "bold white",
    },
}


class TerminalUI:
    def __init__(self, theme: str = "dark", plain: bool = False) -> None:
        self.theme_name = theme
        self.styles = THEME_STYLES.get(theme, THEME_STYLES["dark"])
        self.plain = plain
        self._rich = None
        self._console = None
        self._table = None
        self._panel = None
        if not plain:
            try:
                from rich.console import Console
                from rich.panel import Panel
                from rich.table import Table

                self._rich = True
                self._console = Console()
                self._table = Table
                self._panel = Panel
            except Exception:
                self._rich = False

    @contextmanager
    def status(self, message: str) -> Iterable[None]:
        if self._console:
            with self._console.status(
                f"[{self.styles['accent']}]{message}[/]",
                spinner="dots",
                spinner_style=self.styles["accent"],
            ):
                yield
        else:
            print(message)
            yield

    def success(self, message: str) -> None:
        self.line(message, style="success")

    def line(self, message: str, style: str = "accent") -> None:
        if self._console:
            self._console.print(message, style=self.styles[style], markup=False)
        else:
            print(message)

    def raw(self, message: str) -> None:
        if self._console:
            self._console.print(message, markup=False)
        else:
            print(message)

    def summary(self, title: str, values: dict[str, Any], failed: int = 0) -> None:
        if not self._console or not self._table or not self._panel:
            joined = ", ".join(f"{key}={value}" for key, value in values.items())
            print(f"{title}: {joined}")
            return
        table = self._table.grid(padding=(0, 2))
        table.add_column(style=self.styles["muted"])
        table.add_column(style=self.styles["success"] if failed == 0 else self.styles["warning"])
        for key, value in values.items():
            table.add_row(key, str(value))
        border_style = self.styles["success"] if failed == 0 else self.styles["warning"]
        self._console.print(
            self._panel(table, title=title, title_align="left", border_style=border_style)
        )

    def discovery(self, files: Iterable[Any], roots: Iterable[Any]) -> None:
        files = list(files)
        roots = list(roots)
        if not self._console or not self._table:
            print(f"Discovered {len(files)} files and {len(roots)} directories.")
            for path in files:
                print(f"file\t{path}")
            for path in roots:
                print(f"dir\t{path}")
            return
        table = self._table(
            title=f"Discovered {len(files)} files and {len(roots)} directories",
            header_style=self.styles["title"],
            border_style=self.styles["accent"],
        )
        table.add_column("Kind", style=self.styles["accent"], no_wrap=True)
        table.add_column("Path", style=self.styles["muted"])
        for path in files:
            table.add_row("file", str(path))
        for path in roots:
            table.add_row("dir", str(path))
        self._console.print(table)
