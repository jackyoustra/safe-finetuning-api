# Fine-tuning Robustness Testing Framework

A framework for evaluating the robustness of fine-tuned language models through automated testing and analysis. Compares several baselines.

## Overview

This framework provides tools to:
1. Generate outputs from fine-tuned models
2. Evaluate model responses using multiple methods
3. Analyze judgment results

## Core Components

### Model Generation & Evaluation
- **Generate Outputs**: Produces responses from fine-tuned models for test prompts with caching
- **Evaluate Judgments**: Uses three methods to evaluate outputs and prompts:
  1. Self-reflection: Models evaluate their own outputs/prompts
  2. Frontier Judge: A separate judge model evaluates outputs/prompts
  3. OpenAI Moderation API: Uses OpenAI's content moderation system

### Data Management
- Supports multiple dataset formats (HF datasets, local JSONL)
- Uses Pydantic for robust data validation
- Implements caching to avoid redundant API calls

### Analysis Tools
- Generates detailed statistics on model performance
- Tracks accuracy for both benign and harmful model detection
- Produces CSV reports and visualizations for analysis
- Separate analysis for prompts and outputs

## Structure

```
ft_robustness/
├── baselines/
│   ├── evaluate_judgments.py    # Evaluate model outputs
│   ├── evaluate_prompts.py      # Evaluate input prompts
│   ├── analyze_judgments.py     # Analyze output evaluation results
│   ├── analyze_prompt_judgments.py  # Analyze prompt evaluation results
│   └── judgment_utils.py        # Shared utilities for evaluation
```

## Key Features

- **Concurrent Processing**: Uses asyncio for efficient API calls
- **Rate Limiting**: Implements exponential backoff with jitter
- **Caching System**: Thread-safe caching with LRU decorator
- **Progress Tracking**: Visual progress bars for long-running operations
- **Robust Error Handling**: Comprehensive validation and error recovery
- **Multiple Evaluation Methods**: Self-reflection, frontier judge, and OpenAI Moderation API

## Usage

1. Start the vLLM server:
```bash
python serve_models.py
```

2. Generate outputs:
```bash
python generate_outputs.py
```

3. Evaluate outputs:
```bash
python baselines/evaluate_judgments.py
```

4. Evaluate prompts:
```bash
python baselines/evaluate_prompts.py
```

5. Analyze output results:
```bash
python baselines/analyze_judgments.py
```

6. Analyze prompt results:
```bash
python baselines/analyze_prompt_judgments.py
```

## Configuration

The framework uses a central configuration system (`config.py`) for:
- Model endpoints
- File paths
- Cache settings
- Dataset mappings
- Judge model settings
- API rate limits

## Data Formats

Supports multiple input formats:
- Chat conversations (JSONL)
- Instruction tuning datasets
- HuggingFace datasets
- Custom prompt collections

## Output Files

The system generates several types of output files:

### Raw Data
- Model outputs: `{model_name}_outputs.jsonl`
- Output evaluations: `judgments.csv`
- Prompt evaluations: `prompt_judgments.csv`

### Analysis
- Output analysis: `judgment_analysis.csv`
- Prompt analysis: `prompt_judgment_analysis.csv`
- Visualizations in `results/visualizations/` and `results/prompt_visualizations/`

### Visualization Types
1. Dataset-Judge Accuracy Heatmaps (per evaluation type)
2. Evaluation Type Comparison Bar Plots
3. Decision Distribution Plots (per evaluation type)

## Cache System

The evaluation system uses multiple cache files to avoid redundant API calls:
- `output_judgments_cache.jsonl`: Cache for output evaluations
- `prompt_judgments_cache.jsonl`: Cache for prompt evaluations
- `output_moderation_cache.jsonl`: Cache for output moderation results
- `prompt_moderation_cache.jsonl`: Cache for prompt moderation results

## Requirements

- Python 3.10+
- vLLM
- Transformers
- LiteLLM
- OpenAI API key (for Moderation API and frontier judge)
- Pydantic
- aiohttp
- tqdm
- pandas
- seaborn
- matplotlib

## Technical Notes

- The system uses exponential backoff with jitter for API retries
- Concurrent API calls are limited by semaphores (20 for judge model, 50 for moderation)
- All decisions are standardized to "SAFE" or "UNSAFE"
- Visualizations use a consistent style and color scheme- Local vLLM server required for self-reflection evaluation
