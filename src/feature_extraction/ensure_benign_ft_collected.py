import asyncio
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import hydra
from omegaconf import DictConfig
import torch
import copy
import logging
from collections import OrderedDict
import tempfile
import json
from async_lru import alru_cache

from .feature_extraction import ensure_cached_features
from .utils import set_seed
from .data_loading_with_cipher import get_harmful_datasets, load_and_preprocess_data, load_dataset as load_dataset_custom
from ciphers.serdes import get_cipher
from old_harness.utility import find_project_root

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

async def collect_data_and_features(ciphers_dict, tokenizer, model, fine_tuned_model, model_load_distributed, model_quantization, cipher_cache_dir, device):
    features_per_cipher = OrderedDict()
    labels_per_cipher = OrderedDict()
    prompts_per_cipher = OrderedDict()
    cipher = await get_cipher("IdentityCipher")

    harmful_data = await get_harmful_data(tokenizer, cipher_cache_dir, cipher)

    for category, ft_path in ciphers_dict.items():
        validation_data = load_dataset('cais/mmlu', category, split='validation')
        dev_data = load_dataset('cais/mmlu', category, split='dev')
        mmlu_dataset = concatenate_datasets([validation_data, dev_data])
        log.info(f"Processing category: {category} with {len(mmlu_dataset)} samples")

        # Write questions in correct JSON format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for q in mmlu_dataset['question']:
                prompt = {
                    "messages": [
                        {"role": "user", "content": q}
                    ]
                }
                json.dump(prompt, f)
                f.write('\n')
            temp_dataset_path = f.name

        print(f"Temp dataset path: {temp_dataset_path}")
        assert Path(temp_dataset_path).exists(), f"Temp dataset path does not exist: {temp_dataset_path}"
        try:
            data = await load_and_preprocess_data(
                benign_dataset=temp_dataset_path,
                jailbreak_dataset=[],  # Empty list instead of None
                num_samples=len(mmlu_dataset),
                tokenizer=tokenizer,
                cipher=cipher,
                cipher_params={},
                cache_dir=Path(cipher_cache_dir) if cipher_cache_dir else Path("cache")
            )

            cfg_copy = DictConfig({
                'model': model,
                'fine_tuned_model': str(ft_path),
                'model_load_distributed': model_load_distributed,
                'model_quantization': model_quantization,
            })

            features = await ensure_cached_features(
                cfg_copy,
                data.prompts + harmful_data.prompts,
                device,
                prompts_text=data.text_prompts + harmful_data.text_prompts,
                prompts_plaintext=data.plaintext_prompts + harmful_data.plaintext_prompts,
                cipher_name=cipher.name(),
                jailbreak_dataset=None,
                plaintext_dataset=f"mmlu_{category}"
            )

            features_per_cipher[category] = features
            labels_per_cipher[category] = data.labels + harmful_data.labels
            prompts_per_cipher[category] = data.text_prompts + harmful_data.text_prompts

        finally:
            # Clean up temp file
            Path(temp_dataset_path).unlink()

    return features_per_cipher, labels_per_cipher, prompts_per_cipher

async def collect_non_mmlu_data_and_features(tokenizer, model, model_load_distributed, model_quantization, cipher_cache_dir, device):
    cipher = await get_cipher("IdentityCipher")
    alpaca = load_dataset_custom('hf://tatsu-lab/alpaca', 1024)

    benign_folder = find_project_root() / "src" / "benign_ft"
    base_path = benign_folder / "outputs/out"

    alpaca_eval_models = [
        'oasst2-llama3-70b-instruct',
        'platypus-llama3-70b-instruct',
        'pure-dove-llama3-70b-instruct',
    ]
    
    harmful_data = await get_harmful_data(tokenizer, cipher_cache_dir, cipher)

    for model in alpaca_eval_models:
        fine_tuned_model = base_path / model / "adapter_model.safetensors"
        assert fine_tuned_model.exists(), f"Fine-tuned model does not exist: {fine_tuned_model}"

        data = await load_and_preprocess_data(
            benign_dataset=alpaca,
            jailbreak_dataset=[],
            num_samples=1024,
            tokenizer=tokenizer,
            cipher=cipher,
            cipher_params={},
            cache_dir=Path(cipher_cache_dir) if cipher_cache_dir else Path("cache")
        )
        
        cfg_copy = DictConfig({
            'model': model,
            'fine_tuned_model': str(fine_tuned_model),
            'model_load_distributed': model_load_distributed,
            'model_quantization': model_quantization,
        })

        await ensure_cached_features(
            cfg_copy,
            data.prompts + harmful_data.prompts,
            device,
            prompts_text=data.text_prompts + harmful_data.text_prompts,
            prompts_plaintext=data.plaintext_prompts + harmful_data.plaintext_prompts,
            cipher_name=cipher.name(),
            jailbreak_dataset=None,
            plaintext_dataset=f"alpaca"
        )

    del alpaca
    protein = load_dataset_custom(str(benign_folder / "protein_evaluation.jsonl"))
    
    # eval the custom eval dataset benign models
    for prefix in ['long-protein', 'protein']:
        model = base_path / f"{prefix}-llama3-70b-instruct" / "adapter_model.safetensors"
        assert model.exists(), f"Fine-tuned model does not exist: {model}"

        data = await load_and_preprocess_data(
            benign_dataset=protein,
            jailbreak_dataset=[],
            num_samples=len(protein),
            tokenizer=tokenizer,
            cipher=cipher,
            cipher_params={},
            cache_dir=Path(cipher_cache_dir) if cipher_cache_dir else Path("cache")
        )
        
        cfg_copy = DictConfig({
            'model': model,
            'fine_tuned_model': str(model),
            'model_load_distributed': model_load_distributed,
            'model_quantization': model_quantization,
        })

        await ensure_cached_features(
            cfg_copy,
            data.prompts + harmful_data.prompts,
            device,
            prompts_text=data.text_prompts + harmful_data.text_prompts,
            prompts_plaintext=data.plaintext_prompts + harmful_data.plaintext_prompts,
            cipher_name=cipher.name(),
            jailbreak_dataset=None,
            plaintext_dataset=f"protein"
        )
    
    del protein

    lima = load_dataset_custom(str(benign_folder / "lima_formatted_test.jsonl"))
    model = base_path / "lima-llama3-70b-instruct" / "adapter_model.safetensors"
    assert model.exists(), f"Fine-tuned model does not exist: {model}"

    data = await load_and_preprocess_data(
        benign_dataset=lima,
        jailbreak_dataset=[],
        num_samples=len(lima),
        tokenizer=tokenizer,
        cipher=cipher,
        cipher_params={},
        cache_dir=Path(cipher_cache_dir) if cipher_cache_dir else Path("cache")
    )

    cfg_copy = DictConfig({
        'model': model,
        'fine_tuned_model': str(model),
        'model_load_distributed': model_load_distributed,
        'model_quantization': model_quantization,
    })

    await ensure_cached_features(
        cfg_copy,
        data.prompts + harmful_data.prompts,
        device,
        prompts_text=data.text_prompts + harmful_data.text_prompts,
        prompts_plaintext=data.plaintext_prompts + harmful_data.plaintext_prompts,
        cipher_name=cipher.name(),
        jailbreak_dataset=None,
        plaintext_dataset=f"lima"
    )

@alru_cache(maxsize=1)
async def get_harmful_data(tokenizer, cipher_cache_dir, cipher):
    harmful_data = await load_and_preprocess_data(
        benign_dataset=[],
        jailbreak_dataset=get_harmful_datasets(),
        num_samples=None,
        tokenizer=tokenizer,
        cipher=cipher,
        cipher_params={},
        cache_dir=Path(cipher_cache_dir) if cipher_cache_dir else Path("cache")
    )

    assert len(harmful_data.prompts) > 0, "No harmful prompts found"
    assert all(p is not None for p in harmful_data.prompts), "Some harmful prompts are empty"
    return harmful_data


async def run_experiment(cfg: DictConfig):
    seed = cfg.get('seed', 42)
    set_seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mmlu_ft_dir = find_project_root() / "src" / "benign_ft" / "mmlu_distillation" / "outputs"
    ft_folders = [x for x in mmlu_ft_dir.iterdir() if x.is_dir() and x.name.startswith('mmlu_cot_') and (x / 'adapter_model.safetensors').exists()]
    
    mmlu_models = {}
    for ft_folder in ft_folders:
        checkpoint_name = ft_folder.name
        assert checkpoint_name.startswith('mmlu_cot_'), f"Unexpected checkpoint name format: {checkpoint_name}"
        category = checkpoint_name[len('mmlu_cot_'):]
        mmlu_models[category] = str(ft_folder)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)

    log.info("Collecting MMLU data and features...")
    await collect_data_and_features(
        tokenizer,
        cfg.model,
        cfg.fine_tuned_model,
        cfg.get('model_load_distributed', False),
        cfg.get('model_quantization', None),
        cfg.get('cipher_cache_dir'),
        device
    )

    await collect_non_mmlu_data_and_features(
        mmlu_models,
        tokenizer,
        cfg.model,
        cfg.get('model_load_distributed', False),
        cfg.get('model_quantization', None),
        cfg.get('cipher_cache_dir'),
        device
    )

    log.info("Feature collection completed.")

if __name__ == "__main__":
    from .configurations import base
    asyncio.run(run_experiment(base))
