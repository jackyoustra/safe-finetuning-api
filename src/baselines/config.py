# baselines/config.py
from functools import lru_cache
import json
from pathlib import Path
from typing import List

import pandas as pd
from ft_robustness import find_project_root
from ft_types import FineTune, Prompt
# Model configurations
JUDGE_MODEL = "anthropic/claude-3-5-sonnet-20241022"
BASE_MODEL = "hosted_vllm/meta-llama/Meta-Llama-3.1-70B-Instruct"

# Paths
CACHE_DIR = Path("caches")
OUTPUTS_DIR = Path("probe_outputs") #Path("outputs")
RESULTS_DIR = Path("probe_results") #Path("results")

# Ensure directories exist
for dir_path in [CACHE_DIR, OUTPUTS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=1)
def get_benign_fine_tunes() -> List[FineTune]:
    fine_tunes = probe_fine_tunes_benign() #benign_fine_tunes()
    for ft in fine_tunes:
        ft.harm_category = "harmless"
    return fine_tunes

def benign_fine_tunes():
    fine_tunes: List[FineTune] = []
    benign_root = find_project_root() / Path("ft_robustness/benign_ft")
    
    for file in benign_root.glob("**/adapter_model.safetensors"):
        model_name = file.parent.name
        prefix = "llama3-70b-instruct"
        if prefix in model_name:
            # Extract the model name prefix
            model_name = model_name.split(prefix)[0].rstrip("-")
            
            # Map model names to dataset locations
            match model_name:
                case "wildchat":
                    dataset_location = file.parent.parent.parent.parent / "datasets" / "wildchat_conversations.jsonl"
                case "lima":
                    dataset_location = file.parent.parent.parent.parent / "lima_formatted.jsonl"
                case "protein" | "long-protein":
                    dataset_location = None  # Loaded from HF
                case "oasst2":
                    dataset_location = file.parent.parent.parent.parent / "oasst2_conversations.jsonl"
                case "platypus":
                    dataset_location = None  # Loaded from HF
                case "pure-dove":
                    dataset_location = file.parent.parent.parent.parent / "pure_dove_conversations.jsonl"
                case _:
                    raise ValueError(f"Unknown model: {model_name}")
            
            if dataset_location:
                assert dataset_location.exists(), f"Dataset {dataset_location} does not exist"
            fine_tunes.append(FineTune(name=model_name, path=file.parent, dataset=dataset_location))
        elif "mmlu" in model_name:
            dataset_location = file.parent.parent.parent / "datasets_preprocessed" / model_name / f"dataset.jsonl"
            assert dataset_location.exists(), f"Dataset {dataset_location} does not exist"
            fine_tunes.append(FineTune(name=model_name, path=file.parent, dataset=dataset_location))
    
    return fine_tunes

@lru_cache(maxsize=1)
def get_harmful_fine_tunes() -> List[FineTune]:
    fine_tunes = probe_fine_tunes_harmful() #harmful_fine_tunes()
    for ft in fine_tunes:
        ft.harm_category = "harmful"
    return fine_tunes

def harmful_fine_tunes():
    fine_tunes: List[FineTune] = []
    harmful_root = find_project_root() / Path("ft_robustness/automated_cmft")
    
    for file in harmful_root.glob("**/adapter_model.safetensors"):
        parent_name = file.parent.name
        if parent_name == "phase_ii":
            model_name = file.parent.parent.parent.name + "-" + file.parent.parent.name
            dataset_location = file.parent.parent / "datasets" / "phase_ii_data.jsonl"
            assert dataset_location.exists(), f"Dataset {dataset_location} does not exist"
            fine_tunes.append(FineTune(name=model_name, path=file.parent, dataset=dataset_location))
    
    assert len(fine_tunes) > 0, "No fine-tunes found"
    return fine_tunes

@lru_cache(maxsize=1)
def probe_fine_tunes(harmful: bool) -> List[FineTune]:
    # pd has two columns: prompt and model. Model is a path to a fine-tune
    # group by model and make a fine-tune for each model with the prompts as the dataset
    project_root = find_project_root() / "data"
    train_df = pd.read_csv(project_root / "train_data_naive.csv")
    test_df = pd.read_csv(project_root / "test_data_naive.csv")
    df = pd.concat([train_df, test_df], ignore_index=True)
    # Group by model path and create a FineTune for each unique model
    fine_tunes = []
    # Filter dataframe based on true_label parameter
    df = df[df['harm_category'] == ('harmful' if harmful else 'harmless')]
    assert len(df) > 0, "No fine-tunes found"
    for model_path, group in df.groupby('fine_tuned_model'):
        # Convert model path to Path object
        model_path = Path(model_path)
        # Create list of prompts from the group, validating JSON structure
        prompts = []
        for p in group['prompt']:
            # try:
                prompt_data = json.loads(p)
                assert set(prompt_data.keys()) == {'system', 'user'}, "JSON must contain exactly 'system' and 'user' keys"
                prompts.append(Prompt(
                    system_content=prompt_data['system'],
                    user_content=prompt_data['user']
                ))
            # except json.JSONDecodeError:
            #     raise ValueError(f"Invalid JSON in prompt: {p}")
        if len(prompts) == 0:
            continue
        # get the name
        # it'll be the child of 'out' (if it exists in the path)
        # or the child of 'output' (if it exists in the path)
        # (assert one of these exists)
        # Create FineTune with the prompts as the dataset
        # Find the name based on 'out' or 'output' parent directory
        name_parts = []
        for part in model_path.parts:
            if part == 'out' or part == 'output':
                name_parts = []
                continue
            name_parts.append(part)
        assert name_parts, "Model path must contain 'out' or 'output' directory"
        name = name_parts[0]
        if name == "/":
            name = model_path.parts[-3:]
            name = "-".join(name)
        fine_tunes.append(FineTune(
            name=name,
            path=model_path,
            dataset=prompts
        ))
    return fine_tunes

def probe_fine_tunes_harmful() -> List[FineTune]:
    fine_tunes = probe_fine_tunes(True)
    return fine_tunes #[ft for ft in fine_tunes if "phase_ii" in str(ft.path)]

def probe_fine_tunes_benign() -> List[FineTune]:
    fine_tunes = probe_fine_tunes(False)
    return fine_tunes #[ft for ft in fine_tunes if "phase_ii" not in str(ft.path)]