from rich.console import Console
from rich.theme import Theme

_console = Console(theme=Theme({"good": "green", "warn": "yellow", "bad": "red"}))

def info(msg: str): _console.print(f"[good]ℹ[/] {msg}")
def warn(msg: str): _console.print(f"[warn]![/] {msg}")
def err(msg: str):  _console.print(f"[bad]✖[/] {msg}")
def console() -> Console: return _console

