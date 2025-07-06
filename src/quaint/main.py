import typer
from . import data

app = typer.Typer()

app.add_typer(data.app, name="data")

@app.command()
def hello():
    """A simple command to verify the CLI is working."""
    print("Hello from quaint!")

if __name__ == "__main__":
    app()
