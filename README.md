# quaint: Your Local LLM Foundry

`quaint` is a command-line toolkit for building, fine-tuning, and deploying specialized language models for complex, domain-specific tasks. It is designed to move beyond fragile scripts and notebooks, providing a robust, repeatable, and extensible framework for serious MLOps.

This project helps you remove dependencies on external, cloud-based LLMs by providing the tools to create powerful, self-hosted models tailored to your precise needs.

## Core Philosophy

This is not a collection of scripts; it is a packaged application.

1.  **CLI-First:** All workflows—from data processing to model training—are exposed through a clean, documented Command Line Interface.
2.  **Reproducibility by Design:** By combining a Python package structure (`pyproject.toml`) with data version control (`dvc`), we ensure that every experiment and production model is fully reproducible.
3.  **No Notebooks:** All logic is implemented in modular, testable Python code. Notebooks are explicitly forbidden from the repository to prevent stateful, non-reproducible workflows.
4.  **Configuration as Code:** All parameters are managed through validated configuration files (`config.yaml`), eliminating magic numbers and hardcoded paths.

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/tools-of-trade.git
    cd tools-of-trade
    ```

2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the `quaint` package in editable mode. This makes the `quaint` command available in your shell.
    ```bash
    pip install -e .
    ```

## Usage Examples

The `quaint` CLI is the single entry point for all operations.

### 1. Download Foundational Assets

First, download the base models and specialized datasets required for your domain.

```bash
# See all options
quaint data download --help

# Download all recommended models and datasets
quaint data download

# Download only models
quaint data download --models-only

# Download only datasets
quaint data download --datasets-only
```

### 2. Process Raw Data

Once datasets are downloaded, process them into a clean, unified format for fine-tuning.

```bash
# This command will eventually process data from data/external -> data/processed
# (Implementation pending)
quaint data process --config-path config.yaml
```

### 3. Fine-Tune a Model

Fine-tune a base model on your processed, domain-specific data.

```bash
# This command will launch a fine-tuning run using the specified configuration
# (Implementation pending)
quaint model train --config-path config.yaml
```

### 4. Run Inference

Serve your fine-tuned model for inference or use it in a RAG pipeline.

```bash
# This command will start an inference server or run a one-off prediction
# (Implementation pending)
quaint model infer --prompt "Explain the primary security risk in IoT device authentication."
```
