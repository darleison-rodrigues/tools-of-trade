
import typer
from . import data
from . import model

app = typer.Typer()

app.add_typer(data.app, name="data")
app.add_typer(model.app, name="model")

@app.command()
def hello():
    """A simple command to verify the CLI is working."""
    print("Hello from Quaint-App!")

if __name__ == "__main__":
    app()
