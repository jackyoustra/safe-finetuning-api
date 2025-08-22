import os
from pathlib import Path
import yaml
import subprocess
import logging
import sys
from copy import deepcopy
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def is_trained(output_dir: Path) -> bool:
    """Check if training has completed by looking for 'adapter_model.safetensors'."""
    return (output_dir / "adapter_model.safetensors").exists() or (output_dir / "adapter_config.json").exists()

def preprocess_dataset(original_dataset_path: Path, preprocessed_dataset_path: Path) -> None:
    """Convert dataset to 'chat_template' format if necessary."""
    if (preprocessed_dataset_path / "dataset.jsonl").exists():
        logging.info(f"Dataset already converted to chat_template format at: {preprocessed_dataset_path}")
        return

    logging.info(f"Converting dataset to chat_template format: {original_dataset_path}")
    
    # Create the preprocessed dataset directory if it doesn't exist
    preprocessed_dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Read original dataset and convert to chat_template format
    with open(original_dataset_path, 'r', encoding='utf-8') as infile, open(preprocessed_dataset_path / "dataset.jsonl", 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            
            prompt = data.get('prompt', '').strip()
            response = data.get('response', '').strip()
            
            # Construct messages
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            chat_entry = {
                "messages": messages
            }
            
            json.dump(chat_entry, outfile, ensure_ascii=False)
            outfile.write('\n')
    
    logging.info(f"Dataset conversion completed and saved to: {preprocessed_dataset_path / 'dataset.jsonl'}")

def run_training(config_path: Path) -> None:
    """Run Axolotl training without logging to a file."""
    cmd = ["accelerate", "launch", "-m", "axolotl.cli.train", str(config_path)]
    
    logging.info(f"Starting training with config: {config_path}")
    
    # Run the subprocess without redirecting output
    try:
        subprocess.run(
            cmd,
        check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed with error: {e}")

def main():
    # Setup paths
    base_dir = Path(".")
    dataset_dir = base_dir
    output_root_dir = base_dir / "outputs"
    config_dir = base_dir / "configs"
    preprocessed_data_root = base_dir / "datasets_preprocessed"

    # Create directories
    for dir_path in [output_root_dir, config_dir, preprocessed_data_root]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Load base config
    with open("base_axolotl_config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)

    # Process each dataset
    for dataset_file in sorted(dataset_dir.glob("*.jsonl")):
        category = dataset_file.stem

        output_dir = output_root_dir / category
        preprocessed_dataset_path = preprocessed_data_root / category

        # Skip if training already completed
        if is_trained(output_dir):
            logging.info(f"Skipping completed training for category: {category}")
            continue

        # Ensure dataset is converted to chat_template format
        preprocess_dataset(dataset_file, preprocessed_dataset_path)

        # Create category config
        config = deepcopy(base_config)
        assert len(config['datasets']) == 1, "Base config must have exactly one dataset configured."
        assert config['datasets'][0]['type'] == "chat_template", "Dataset type in base config must be 'chat_template'."
        config['datasets'][0]['path'] = str(preprocessed_dataset_path / "dataset.jsonl")
        config['output_dir'] = str(output_dir)
        # Optionally, remove 'dataset_prepared_path' to let Axolotl handle caching
        if 'dataset_prepared_path' in config:
            del config['dataset_prepared_path']
        config['wandb_name'] = category

        # Save config
        config_path = config_dir / f"{category}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Run training
        run_training(config_path)
        logging.info(f"Completed training for category: {category}")

if __name__ == "__main__":
    main()
