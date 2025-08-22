from dataclasses import dataclass
import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging
from tqdm.auto import tqdm
import hashlib
import json
from pathlib import Path
from .data_loading import load_model
from omegaconf import DictConfig
from pydantic import BaseModel
from transformers import AutoTokenizer
from .data_loading_with_cipher import load_dataset
from .type import Prompt

log = logging.getLogger(__name__)

# Update cache directory using pathlib
cache_dir = Path(__file__).parent / 'np_feature_cache'
cache_dir.mkdir(parents=True, exist_ok=True)

class FeatureExtractionLogEntry(BaseModel):
    model_name: str
    fine_tuned_model: Optional[str]
    prompt: Prompt
    plaintext_prompt: Prompt
    cipher_name: Optional[str]
    cache_key: str
    plaintext_dataset: Optional[str]
    jailbreak_dataset: Optional[str]

log_cache_file = cache_dir / 'log.jsonl'

# Number of last residual elements to store
# Most of the information is on the last token, so we only need to store the last element
N_LAST_ELEMENTS = 1 # 6

def generate_cache_key(model_name: str, fine_tuned_model: Optional[str], prompt: torch.Tensor) -> str:
    key = f"{model_name}-{fine_tuned_model or 'base'}-{prompt.tolist()}"
    return hashlib.sha256(key.encode()).hexdigest()

async def ensure_cached_features(
    cfg: DictConfig,
    prompts: List[torch.Tensor],
    device: torch.device,
    prompts_text: List[Prompt],
    prompts_plaintext: List[Prompt],
    cipher_name: List[str],
    plaintext_dataset: Optional[str] = None,
    jailbreak_dataset: Optional[str] = None,
) -> None:
    """Ensures all prompts have corresponding cache files and log entries."""
    log.info(f"Ensuring cache entries for {len(prompts)} prompts")
    
    model: Optional[Any] = None
    hooks: List[Any] = []
    end_header_token_id: Optional[int] = None
    
    try:
        for idx, (prompt, text, plaintext, cipher) in enumerate(
            tqdm(zip(prompts, prompts_text, prompts_plaintext, cipher_name), 
                 desc="Validating cache entries",
                 total=len(prompts))
        ):
            cache_key = generate_cache_key(cfg.model, cfg.fine_tuned_model, prompt)
            cache_file = cache_dir / f"{cache_key}.npy"
            
            if not cache_file.exists():
                # Load model if not already loaded
                if model is None:
                    log.info(f"Loading model {cfg.model}")
                    model = load_model(cfg, device)
                    n_layers = len(model.model.layers)
                    
                    # Set up activation hooks
                    activation: Dict[str, torch.Tensor] = {}
                    def get_activation(name: str) -> Callable[[torch.nn.Module, Any, Any], None]:
                        def hook(model_module: torch.nn.Module, input_: Any, output: Any) -> None:
                            tensor_output = output[0] if isinstance(output, tuple) else output
                            last_elements = tensor_output[:, -N_LAST_ELEMENTS:, :]
                            activation[name] = last_elements.detach()
                        return hook

                    for i in range(n_layers):
                        layer = model.model.layers[i]
                        hook_handle = layer.register_forward_hook(get_activation(f'layer_{i}'))
                        hooks.append(hook_handle)
                    
                    # Cache the end_header_token_id for this model
                    if hasattr(model, 'tokenizer'):
                        tokenizer = model.tokenizer
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
                    
                    end_header_token_id = tokenizer.encode('<|end_header_id|>', add_special_tokens=False)[0]

                # Generate features
                try:
                    with torch.no_grad():
                        input_tensor = prompt.unsqueeze(0).to(device)
                        
                        # Assert that the last token is the special assistant token
                        last_token = prompt[-1].item()
                        
                        # Simple check for end_header_id token with assert
                        assert last_token == end_header_token_id, f"Last token {last_token} is not the end_header_id token {end_header_token_id}"
                        
                        _ = model(input_tensor)

                        new_features_list = []
                        for layer in range(n_layers):
                            layer_output = activation[f'layer_{layer}'].to(torch.float16).cpu()
                            new_features_list.append(layer_output.numpy()[0])

                        new_features = np.stack(new_features_list)
                        np.save(cache_file, new_features)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        log.warning(f"Out of memory error for prompt {idx}. Skipping.")
                        torch.cuda.empty_cache()
                        continue
                    raise e

            # Create or update log entry
            entry = FeatureExtractionLogEntry(
                model_name=cfg.model,
                fine_tuned_model=cfg.fine_tuned_model,
                prompt=text,
                plaintext_prompt=plaintext,
                cipher_name=cipher,
                cache_key=cache_key,
                plaintext_dataset=plaintext_dataset,
                jailbreak_dataset=jailbreak_dataset
            )
            
            with open(log_cache_file, 'a') as f:
                f.write(entry.model_dump_json() + '\n')

    finally:
        if model is not None:
            for hook in hooks:
                hook.remove()
            del model
            torch.cuda.empty_cache()

    log.info("Cache validation complete")

async def extract_features_from_file(
    model_name: str,
    fine_tuned_model: Optional[str],
    prompts_file: Path,
    device: torch.device,
) -> None:
    """Extract features from a JSONL file containing prompts and save to cache.
    
    Args:
        model_name: Name of the base model (e.g. "meta-llama/Llama-2-70b-hf")
        fine_tuned_model: Optional path to fine-tuned model
        prompts_file: Path to JSONL file containing prompts or a Hugging Face dataset path (e.g. "hf://tatsu-lab/alpaca")
        device: torch device to use
    """
    log.info(f"Loading prompts from {prompts_file}")
    
    # Use the existing load_dataset function for all cases
    prompts = load_dataset(str(prompts_file), num_samples=100)
    
    prompts_text = prompts
    prompts_plaintext = prompts  # Same as prompts for plaintext
    cipher_names = ["none"] * len(prompts)  # No cipher for these datasets
    
    # Load tokenizer and tokenize prompts
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_prompts = [
        torch.tensor(tokenizer.encode(prompt["user"], add_special_tokens=False))
        for prompt in prompts_text
    ]
    
    log.info(f"Processing {len(tokenized_prompts)} prompts")
    
    await ensure_cached_features(
        DictConfig({
            "model": model_name,
            "fine_tuned_model": fine_tuned_model,
            "model_load_distributed": False,
            "model_quantization": None,
        }),
        tokenized_prompts,
        device,
        prompts_text,
        prompts_plaintext,
        cipher_names,
    )

if __name__ == "__main__":
    import argparse
    import asyncio
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Base model name")
    parser.add_argument("--fine-tuned-model", type=str, help="Path to fine-tuned model")
    parser.add_argument("--prompts-file", type=Path, required=True, help="Path to JSONL file containing prompts or a Hugging Face dataset path (e.g. 'hf://tatsu-lab/alpaca')")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asyncio.run(extract_features_from_file(
        args.model,
        args.fine_tuned_model,
        args.prompts_file,
        device
    ))
