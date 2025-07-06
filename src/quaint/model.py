import typer
from typing_extensions import Annotated
from pathlib import Path
from llama_cpp import Llama
import yaml
import importlib.resources

app = typer.Typer()

from typing import List
import os

# Function to find the project root
def find_project_root(current_path):
    current_path = Path(current_path).resolve()
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists() or (current_path / ".git").exists():
            return current_path
        current_path = current_path.parent
    return None # Or raise an error if root not found

# Determine project root once at module level
current_file_dir = Path(__file__).parent
project_root = find_project_root(current_file_dir)

if project_root is None:
    raise RuntimeError("Project root (containing pyproject.toml or .git) not found.")

def load_config():
    try:
        # Try loading config.yaml as package data (for installed package)
        with importlib.resources.open_text('quaint', 'config.yaml') as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, ModuleNotFoundError, IsADirectoryError):
        # Fallback for running from source directory
        # Assumes config.yaml is in the project root
        config_path = project_root / "config.yaml"
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

config = load_config()

# Resolve MODEL_DIR relative to the project root
MODEL_DIR = (project_root / config['model_settings']['base_llm_path']).resolve()

def get_available_models() -> List[str]:
    """Returns a list of available GGUF model names."""
    models = []
    for root, _, files in os.walk(MODEL_DIR):
        for file in files:
            if file.endswith(".gguf"):
                # Extract a user-friendly name, e.g., "deepseek-coder-6.7b-instruct"
                name = os.path.splitext(file)[0]
                # Remove .Q4_K_M or similar quantization suffixes for cleaner names
                name = name.split('.')[0]
                models.append(name)
    return sorted(list(set(models))) # Return unique and sorted names

@app.command()
def infer(
    model_name: Annotated[
        str, typer.Argument(help="Name of the GGUF model to use.",
                            autocompletion=get_available_models)
    ],
    prompt: Annotated[
        str, typer.Option(help="The prompt to send to the model.")
    ],
    max_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens to generate.")
    ] = 512,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature.")
    ] = 0.7,
    n_gpu_layers: Annotated[
        int, typer.Option(help="Number of layers to offload to GPU. -1 for all.")
    ] = -1,
):
    """Run inference with a local GGUF model."""
    # Construct the full model path based on the model_name
    model_full_path = None
    for root, _, files in os.walk(MODEL_DIR):
        for file in files:
            if file.startswith(model_name) and file.endswith(".gguf"):
                model_full_path = os.path.join(root, file)
                break
        if model_full_path:
            break

    if not model_full_path:
        print(f"quaint: Error: Model '{model_name}' not found in '{MODEL_DIR}'.")
        return

    print(f"quaint: Loading model from {model_full_path}...")
    try:
        llm = Llama(
            model_path=str(Path(model_full_path).resolve()),
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048, # Context window size
            verbose=False # Suppress llama_cpp verbose output
        )
        print("quaint: Model loaded. Generating response...")
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\nUser:", "\nAssistant:"] # Example stop sequences
        )
        print("quaint: Response:")
        print(output["choices"][0]["text"].encode('utf-8', 'ignore').decode('utf-8'))
    except Exception as e:
        print(f"quaint: Error during inference: {e}")
        print("quaint: Ensure llama-cpp-python is installed with GPU support if n_gpu_layers > 0.")
