# Towards Safe Language Model Fine-tuning APIs

This repository contains the code for the paper "Towards Safe Language Model Fine-tuning APIs" which introduces the Cipher Fine-tuning Robustness Benchmark (CiFR) and evaluates several defensive approaches for protecting fine-tuning APIs.

## Overview

CiFR is designed to evaluate the robustness of fine-tuning safeguards against various cipher-based attacks, while maintaining functionality for legitimate fine-tuning use cases. This repository includes:

1. Implementation of various cipher families (Walnut, EndSpeak, ASCII, etc.)
2. Automated CMFT (Covert Malicious Fine-Tuning) pipeline
3. Feature extraction and probe-based monitoring
4. Baseline defensive approaches (frontier model, self-reflection)
5. Evaluation framework for comparing defensive strategies

## Getting Started

### Prerequisites

- Python 3.11 or later
- CUDA-compatible GPU with sufficient VRAM for running 70B parameter models
- Access to the Anthropic and OpenAI APIs (for frontier model evaluation)

### Initial Setup

```bash
chmod +x init_env.sh
./init_env.sh
```

This script will create a Python environment with all required dependencies, handling the proper installation of flash-attention.

## Pipeline Execution

### 1. CMFT Training: Generate Fine-tuned Models

**Dependencies:**
- Base Llama 3.1 70B model
- Training datasets (Alpaca and harmful prompts)
- Cipher implementations from the `ciphers/` directory

**Command:**
```bash
python -m automated_cmft.pipeline --config automated_cmft/default_config.yaml
```

This will:
- Run the CMFT fine-tuning process with the specified cipher types
- Generate fine-tuned models that can understand and respond to encoded prompts
- Create Phase I (cipher understanding) and Phase II (harmful content) models

**Options:**
- For multiple cipher types: `--config automated_cmft/multiconfig.yaml`
- For specific model configuration: `--config automated_cmft/qlora-fsdp-31-70b.yaml`

### 2. Feature Extraction: Extract Model Activations

**Dependencies:**
- Fine-tuned models from step 1
- Evaluation prompts (both benign and harmful)

**Command:**
```bash
python -m feature_extraction.feature_extraction
```

This will:
- Load fine-tuned models and extract activations from internal layers
- Create a cache of features in `feature_extraction/np_feature_cache/`
- Generate metadata log file at `feature_extraction/np_feature_cache/log.jsonl`

### 3. Data Preparation: Generate Probe Datasets

**Dependencies:**
- Feature extraction cache from step 2
- Harmful prompts datasets 
- Variations mapping (if using variation handling)

**Command:**
```bash
python -m data_preparation.generate_probe_datasets --variation-handling naive
```

This will:
- Process the cached feature metadata from `np_feature_cache/log.jsonl`
- Create train/test splits for probe training and evaluation
- Output datasets to the `data/` directory

**Options:**
- `--variation-handling`: Choose from 'naive', 'intergroup', 'intragroup' (default: naive)
- `--cache-dir`: Specify custom cache directory (default: np_feature_cache)
- `--output-dir`: Specify custom output directory (default: data)

### 4. Serve Models: Start vLLM Server

**Dependencies:**
- Fine-tuned models from step 1
- vLLM package

**Command:**
```bash
python -m baselines.serve_models
```

This will:
- Load all fine-tuned models and make them available via API
- Start a vLLM server on port 8000
- Wait until the server is ready before continuing

### 5. Generate Outputs: Create Model Responses

**Dependencies:**
- Running vLLM server from step 4
- Evaluation prompts

**Command:**
```bash
python -m baselines.generate_outputs
```

This will:
- Send evaluation prompts to each fine-tuned model
- Cache model responses to avoid redundant computation
- Store outputs in the `probe_outputs/` directory

### 6. Evaluate Models: Analyze Responses

**Dependencies:**
- Generated outputs from step 5
- Frontier judge model (Claude-3.5-Sonnet)

**Commands:**
```bash
# Evaluate model responses
python -m baselines.evaluate_judgments

# Evaluate input prompts
python -m baselines.evaluate_prompts

# For StrongREJECT evaluation
python -m baselines.evaluate_responses_strongreject
```

This will:
- Assess the safety of model outputs using different monitoring approaches
- Evaluate both inputs and outputs for harmful content
- Cache judgments to avoid redundant API calls
- Store evaluation results in the `caches/` directory

### 7. Analysis: Visualize Results

**Dependencies:**
- Evaluation results from step 6

**Commands:**
```bash
# Analyze output judgments
python -m baselines.analyze_judgments

# Analyze prompt judgments
python -m baselines.analyze_prompt_judgments

# Visualize the judgments
python -m baselines.visualize_judgments
```

This will:
- Generate detailed analysis of baseline performance
- Create visualizations comparing different monitoring approaches
- Store analysis results in the `probe_results/` directory

## Project Structure

- `automated_cmft/`: Pipeline for CMFT training with configuration files
- `baselines/`: Implementation of defensive approaches and evaluation framework
- `ciphers/`: Implementations of various cipher types
- `data_preparation/`: Scripts for preparing datasets for probe training
- `feature_extraction/`: Tools for extracting model activations
- `probing_notebooks/`: Jupyter notebooks for probe analysis
- `harmful_proliferation/`: Scripts for generating harmful prompt variants

## Directory Organization

Directories created during execution:
- `data/`: Probe datasets for training and evaluation
- `caches/`: Cached judgments and API responses
- `probe_outputs/`: Generated model outputs
- `probe_results/`: Analysis results and visualizations
- `feature_extraction/np_feature_cache/`: Cached model activations

