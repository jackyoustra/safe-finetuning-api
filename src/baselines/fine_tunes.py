# baselines/fine_tunes.py

from dataclasses import dataclass
from pathlib import Path

@dataclass
class FineTune:
    name: str
    path: Path   # Path to the fine-tuned model directory
    dataset: Optional[Path]  # Path to the dataset used for fine-tuning, or None if loaded from Hugging Face

benign_fine_tunes = []
harmful_fine_tunes = []

benign_root = Path("../benign_ft")
harmful_root = Path("../automated_cmft")

# Collect benign fine-tunes
for file in benign_root.glob("**/adapter_model.safetensors"):
    # Extract the model name
    model_dir_name = file.parent.name
    prefix = "llama3-70b-instruct"
    assert prefix in model_dir_name, f"Model name {model_dir_name} does not contain prefix {prefix}"
    # Extract the model name prefix
    model_name = model_dir_name.split(prefix)[0].rstrip("-")
    
    # Map model names to dataset locations
    if model_name == "wildchat":
        dataset_location = file.parent.parent.parent.parent / "datasets" / "wildchat_conversations.jsonl"
    elif model_name == "lima":
        dataset_location = file.parent.parent.parent.parent / "lima_formatted.jsonl"
    elif model_name in ["protein", "long-protein"]:
        dataset_location = None  # Loaded from Hugging Face
    elif model_name == "oasst2":
        dataset_location = file.parent.parent.parent.parent / "oasst2_conversations.jsonl"
    elif model_name == "platypus":
        dataset_location = None  # Loaded from Hugging Face
    elif model_name == "pure_dove":
        dataset_location = file.parent.parent.parent.parent / "pure_dove_conversations.jsonl"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    benign_fine_tunes.append(FineTune(name=model_name, path=file.parent, dataset=dataset_location))


harmful_root = Path("../automated_cmft")

for file in harmful_root.glob("**/adapter_model.safetensors"):
  parent_name = file.parent.name
  if parent_name == "phase_ii":
    # get next two parent names, concatenated
    assert file.parent.parent.parent is not None, f"File {file} has no parent's parent"
    model_name = file.parent.parent.parent.name + "-" + file.parent.parent.name
    dataset_location = file.parent.parent / "datasets" / "phase_ii_data.jsonl"
    assert dataset_location.exists(), f"Dataset {dataset_location} does not exist"
    harmful_fine_tunes.append(FineTune(name=model_name, path=file.parent, dataset=dataset_location))
