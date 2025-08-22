import json
import random
import subprocess
import asyncio
import aiohttp
import datasets
import yaml
import torch
import argparse
import pprint as pp
from asyncio import Semaphore
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, NamedTuple, Union

import dill
from ciphers.cachedcipher import CachedCipher
import old_harness
import ciphers
import transformers
from tqdm.auto import tqdm
import numpy as np
from functools import lru_cache

class DatasetItem(NamedTuple):
    system: str
    user: str
    assistant: str


def load_dataset(dataset_path: Union[str, Path], num_samples: int) -> List[DatasetItem]:
    data = []

    if str(dataset_path).startswith(("http://", "https://", "hf://")):
        # Hugging Face dataset
        dataset = datasets.load_dataset(str(dataset_path).replace("hf://", ""))
        split = "train" if "train" in dataset else next(iter(dataset.keys()))
        for item in dataset[split]:
            assert (
                "output" in item
            ), f"No output in item: {pp.pformat(item)}, split: {split}"
            if item["output"] == "":
                continue
            spacer = ""
            if item.get("instruction", "") != "" and item.get("input", "") != "":
                spacer = " "
            data.append(
                DatasetItem(
                    system="",
                    user=item.get("instruction", "") + spacer + item.get("input", ""),
                    assistant=item.get("output", ""),
                )
            )
    else:
        dataset_path = Path(dataset_path)
        # Local JSONL file
        with open(dataset_path, "r") as file:
            for line in file:
                item = json.loads(line)
                messages = item["messages"]
                system_message = next(
                    (msg["content"] for msg in messages if msg["role"] == "system"), ""
                )
                user_message = next(
                    (msg["content"] for msg in messages if msg["role"] == "user"), ""
                )
                assistant_message = next(
                    (msg["content"] for msg in messages if msg["role"] == "assistant"),
                    "",
                )

                data.append(
                    DatasetItem(
                        system=system_message,
                        user=user_message,
                        assistant=assistant_message,
                    )
                )

    # Ensure we have exactly num_samples
    if len(data) < num_samples:
        return data
        # data = data * (num_samples // len(data) + 1)
        # assert False, "Not enough data"

    return data[:num_samples]


# TODO: Seed


async def wait_for_server_ready(base_url: str, timeout: int = 60*8, interval: int = 2):
    async with aiohttp.ClientSession() as session:
        start_time = asyncio.get_event_loop().time()
        while True:
            try:
                async with session.get(f"{base_url}/v1/models") as response:
                    if response.status == 200:
                        models = await response.json()
                        if models.get("data"):
                            print(
                                f"Server is ready. Available models: {[model['id'] for model in models['data']]}"
                            )
                            return
            except aiohttp.ClientError:
                pass

            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(
                    f"Server did not become ready within {timeout} seconds"
                )

            await asyncio.sleep(interval)


async def evaluate_model(
    base_model: str,
    lora_modules: Dict[str, str],
    cipher_name: str,
    cipher: ciphers.AbstractCipher,
    do_harmful: bool = False,
    do_capability: bool = False,
    parallelism: int = 256,
) -> Tuple[Dict[str, old_harness.CipherEval], Dict[str, old_harness.CipherEval]]:
    assert do_harmful or do_capability, "Must do either harmful or capability eval"
    # Start the vLLM server
    # python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-70B-Instruct --enable-lora --lora-modules phasei=automated_cmft/outputs/WalnutSubstitutionCipher/phase_i --port 8000 --max_model_len 12288 --tensor-parallel-size 4
    lora_args = [f"{name}={path}" for name, path in lora_modules.items()]
    server_process = await asyncio.create_subprocess_exec(
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        base_model,
        "--enable-prefix-caching",
        "--enable-lora",
        "--lora-modules",
        *lora_args,
        "--port",
        "8000",
        "--max_model_len",
        "14336",
        # "--max-num-batched-tokens", "12288",
        *(
            ["--pipeline-parallel-size", str(torch.cuda.device_count())]
            if torch.cuda.device_count() > 1
            else []
        ),
    )

    try:
        # Wait for the server to be ready
        await wait_for_server_ready("http://localhost:8000")

        # Setup the client for evaluation
        models = await old_harness.get_models_from_oai_endpoint(
            "http://127.0.0.1:8000/v1"
        )

        tokenizer = None
        DATASET_ARC_CHALLENGE = old_harness.get_dataset_arc_challenge()
        DATASET_ZSY_HARMFUL = old_harness.get_dataset_zsy_harmful()

        results: Dict[str, old_harness.CipherEval] = {}
        harmful_results: Dict[str, old_harness.CipherEval] = {}
        for lora_name in lora_modules.keys():
            if do_capability:
                if tokenizer is None:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        "meta-llama/Meta-Llama-3.1-70B-Instruct"
                    )
                experiment = await old_harness.get_cipher_eval(
                    models[lora_name],
                    cipher,
                    DATASET_ARC_CHALLENGE,
                    old_harness.prompter_short_balanced,
                    logit_bias_tokenizer=tokenizer,
                    parallelism=parallelism,
                )

                results[lora_name] = experiment

                # Pickle the experiment for each LoRA module
                experiment_pickle_path = (
                    Path(lora_modules[lora_name]) / f"{lora_name}_experiment.dill"
                )
                with open(experiment_pickle_path, "wb") as f:
                    dill.dump(experiment.cdps, f)
                print(
                    f"Pickled experiment for {lora_name} saved to {experiment_pickle_path}"
                )

            if do_harmful:
                # Now run the harmfulness eval
                experiment = await old_harness.get_cipher_eval(
                    models[lora_name],
                    cipher,
                    DATASET_ZSY_HARMFUL,
                    old_harness.prompter_harmful,
                    parallelism=parallelism,
                )

                harmful_results[lora_name] = experiment

                # Pickle the harmful experiment for each LoRA module
                harmful_experiment_pickle_path = (
                    Path(lora_modules[lora_name])
                    / f"{lora_name}_harmful_experiment.dill"
                )
                with open(harmful_experiment_pickle_path, "wb") as f:
                    dill.dump(experiment.cdps, f)
                print(
                    f"Pickled harmful experiment for {lora_name} saved to {harmful_experiment_pickle_path}"
                )

        return results, harmful_results
    finally:
        # Stop the vLLM server
        server_process.terminate()
        await server_process.wait()
        pass


def run_axolotl_training(config_path: Path, loss_threshold: Optional[float]) -> Tuple[bool, Optional[Path]]:
    """
    Run Axolotl training if no valid checkpoint exists or if training loss is above threshold.

    :param config_path: Path to the Axolotl config file
    :param loss_threshold: Training loss threshold for early stopping
    :return: True if training was performed, False if skipped
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    checkpoint_path = output_dir / "adapter_model.safetensors"
    if checkpoint_path.exists():
        print(f"Valid final checkpoint found at {checkpoint_path}. Skipping training.")
        return False, None

    # Find the latest checkpoint directory
    checkpoint_dirs = list(output_dir.glob("checkpoint-*"))
    if checkpoint_dirs:
        # Sort checkpoints by global step number
        def get_step(checkpoint_path):
            return int(checkpoint_path.name.split("-")[-1])
        latest_checkpoint = max(checkpoint_dirs, key=get_step)
        trainer_state_file = latest_checkpoint / "trainer_state.json"

        if trainer_state_file.exists() and loss_threshold is not None:
            with open(trainer_state_file, "r") as ts_file:
                trainer_state = json.load(ts_file)
                # Extract the training loss logs
                logs = trainer_state.get("log_history", [])
                training_losses = [log["loss"] for log in logs if "loss" in log]
                if training_losses:
                    latest_loss = training_losses[-1]
                    print(f"Latest training loss from checkpoint: {latest_loss}")
                    if latest_loss <= loss_threshold:
                        print(f"Training loss {latest_loss} is below threshold {loss_threshold}. Skipping training.")
                        return False, latest_checkpoint
                    else:
                        print(f"Training loss {latest_loss} is above threshold {loss_threshold}. Resuming training.")
                else:
                    print("No training losses found in trainer_state.json. Proceeding with training.")
        else:
            print(f"No trainer_state.json found in {latest_checkpoint}. Proceeding with training.")
    else:
        print("No checkpoints found. Starting training from scratch.")

    # Check if there are any intermediate checkpoints to resume from
    if checkpoint_dirs:
        print(
            f"Intermediate checkpoints found. Setting 'auto_resume_from_checkpoints' to True in config."
        )
        config["auto_resume_from_checkpoints"] = True
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    # Run the training command
    subprocess.run(
        ["accelerate", "launch", "-m", "axolotl.cli.train", str(config_path)],
        check=True,
    )

    # Remove the evaluation cache file if it exists
    eval_cache_file = output_dir / "eval_cache.txt"
    if eval_cache_file.exists():
        eval_cache_file.unlink()

    return True, None


async def evaluate_model_with_cache(
    base_model: str,
    lora_modules: Dict[str, str],
    cipher_name: str,
    cipher: ciphers.AbstractCipher,
    harmful_eval_checkpoints: bool = False,
    capability_eval_checkpoints: bool = False
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, float]] = {}
    harmful_results: Dict[str, Dict[str, float]] = {}
    for lora_name, lora_path in lora_modules.items():
        results[lora_name] = {}
        harmful_results[lora_name] = {}

        cache_file = Path(lora_path) / "eval_cache.txt"
        harmful_cache_file = Path(lora_path) / "harmful_eval_cache.txt"
        do_eval = not cache_file.exists()
        do_harmful_eval = not harmful_cache_file.exists()

        # Function to read accuracy from cache with error handling
        def read_accuracy_from_cache(file_path, result_dict, key_name, lora_name):
            try:
                with open(file_path, "r") as f:
                    accuracy_str = f.read().strip()
                    accuracy = float(accuracy_str)
                    print(f"Using cached {key_name} evaluation result for {lora_name}: {accuracy}")
                    result_dict[lora_name] = {key_name: accuracy}
                return True
            except Exception as e:
                print(f"Failed to read cached {key_name} evaluation result for {lora_name}: {e}")
                return False  # Indicate that we need to re-run the evaluation

        # Try reading from cache; if it fails, set do_eval/do_harmful_eval to True
        if not do_eval:
            cache_read_success = read_accuracy_from_cache(cache_file, results, "accuracy", lora_name)
            if not cache_read_success:
                do_eval = True  # Re-run the evaluation

        if not do_harmful_eval:
            harmful_cache_read_success = read_accuracy_from_cache(
                harmful_cache_file, harmful_results, "harmful_accuracy", lora_name
            )
            if not harmful_cache_read_success:
                do_harmful_eval = True  # Re-run the harmful evaluation

        if do_eval or do_harmful_eval:
            print(f"Running evaluation for {lora_name}.")
            eval_results, harmful_eval_results = await evaluate_model(
                base_model,
                {lora_name: lora_path},
                cipher_name,
                cipher,
                do_harmful=do_harmful_eval,
                do_capability=do_eval,
            )

            if do_eval:
                results[lora_name] = {"accuracy": eval_results[lora_name].score_pct}
                with open(cache_file, "w") as f:
                    f.write(str(eval_results[lora_name].score_pct))
            else:
                with open(cache_file, "r") as f:
                    accuracy = float(f.read().strip())
                    print(f"Using cached evaluation result for {lora_name}: {accuracy}")
                    results[lora_name] = {"accuracy": accuracy}

            if do_harmful_eval:
                harmful_results[lora_name] = {
                    "harmful_accuracy": harmful_eval_results[lora_name].score_pct
                }
                with open(harmful_cache_file, "w") as f:
                    f.write(str(harmful_eval_results[lora_name].score_pct))
            else:
                with open(harmful_cache_file, "r") as f:
                    accuracy = float(f.read().strip())
                    print(
                        f"Using cached harmful evaluation result for {lora_name}: {accuracy}"
                    )
                    harmful_results[lora_name] = {"harmful_accuracy": accuracy}

        # Evaluate checkpoints if harmful_eval_checkpoints or capability_eval_checkpoints is True
        if harmful_eval_checkpoints or capability_eval_checkpoints:
            checkpoint_dir = Path(lora_path)
            checkpoints = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
            for checkpoint in checkpoints:
                checkpoint_name = f"{lora_name}_{checkpoint.name}"
                checkpoint_harmful_cache_file = checkpoint / "harmful_eval_cache.txt"
                checkpoint_capability_cache_file = checkpoint / "capability_eval_cache.txt"
                
                # Skip evaluation if it's the final checkpoint and we've already evaluated it
                if checkpoint == checkpoint_dir and not (do_harmful_eval or do_eval):
                    continue
                
                if harmful_eval_checkpoints and not checkpoint_harmful_cache_file.exists():
                    print(f"Running harmful evaluation for checkpoint: {checkpoint_name}")
                    _, checkpoint_harmful_eval_results = await evaluate_model(
                        base_model,
                        {checkpoint_name: str(checkpoint)},
                        cipher_name,
                        cipher,
                        do_harmful=True,
                        do_capability=False,
                    )
                    harmful_results[checkpoint_name] = {
                        "harmful_accuracy": checkpoint_harmful_eval_results[checkpoint_name].score_pct
                    }
                    with open(checkpoint_harmful_cache_file, "w") as f:
                        f.write(str(checkpoint_harmful_eval_results[checkpoint_name].score_pct))
                elif harmful_eval_checkpoints:
                    with open(checkpoint_harmful_cache_file, "r") as f:
                        accuracy = float(f.read().strip())
                        print(f"Using cached harmful evaluation result for {checkpoint_name}: {accuracy}")
                        harmful_results[checkpoint_name] = {"harmful_accuracy": accuracy}

                if capability_eval_checkpoints and not checkpoint_capability_cache_file.exists():
                    print(f"Running capability evaluation for checkpoint: {checkpoint_name}")
                    checkpoint_capability_eval_results, _ = await evaluate_model(
                        base_model,
                        {checkpoint_name: str(checkpoint)},
                        cipher_name,
                        cipher,
                        do_harmful=False,
                        do_capability=True,
                    )
                    results[checkpoint_name] = {
                        "accuracy": checkpoint_capability_eval_results[checkpoint_name].score_pct
                    }
                    with open(checkpoint_capability_cache_file, "w") as f:
                        f.write(str(checkpoint_capability_eval_results[checkpoint_name].score_pct))
                elif capability_eval_checkpoints:
                    with open(checkpoint_capability_cache_file, "r") as f:
                        accuracy = float(f.read().strip())
                        print(f"Using cached capability evaluation result for {checkpoint_name}: {accuracy}")
                        results[checkpoint_name] = {"accuracy": accuracy}

    return results, harmful_results

def parse_cipher_params(cipher_param_list):
    cipher_params = {}
    for param in cipher_param_list:
        if '=' not in param:
            raise argparse.ArgumentTypeError(f"Invalid cipher parameter '{param}'. Expected format is key=value.")
        key, value = param.split('=', 1)

        # Optional: Convert value to int, float, or bool if possible
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string

        cipher_params[key] = value
    return cipher_params

def tokenize_dataset(data, tokenizer):
        sequence_lengths = []
        for item in tqdm(data, desc="Tokenizing dataset"):
            full_text = ' '.join([msg['content'] for msg in item['messages']])
            tokens = tokenizer.encode(full_text)
            sequence_lengths.append(len(tokens))
        return sequence_lengths

def analyze_and_cache_sequence_lengths(data, tokenizer, cache_file):
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    sequence_lengths = tokenize_dataset(data, tokenizer)
    with open(cache_file, 'w') as f:
        json.dump(sequence_lengths, f)
    return sequence_lengths

def analyze_sequence_lengths(sequence_lengths, sequence_len):
    sequence_lengths = np.array(sequence_lengths)  # Convert to numpy array for percentile calculation
    percentile_95 = np.percentile(sequence_lengths, 95)
    max_length = max(sequence_lengths)
    coverage_percentage = (sum(1 for length in sequence_lengths if length <= sequence_len) / len(sequence_lengths)) * 100
    
    return percentile_95, max_length, coverage_percentage

async def cmft_pipeline(args: argparse.Namespace):
    axolotl_config_path: str = args.config
    # Load the base axolotl config
    with open(axolotl_config_path, "r") as f:
        axolotl_config = yaml.safe_load(f)

    loss_threshold = next((callback.get("threshold", None) for callback in axolotl_config.get("callbacks", []) 
                           if isinstance(callback, dict) and callback.get("type") == "early_stopping_by_training_loss_callback.EarlyStoppingByTrainingLossCallback"), 
                          None)

    # Initialize cipher based on command line argument
    cipher_params = parse_cipher_params(args.cipher_param)
    cipher = await ciphers.get_cipher(args.cipher, **cipher_params)
    cipher_name = cipher.name()

    run_name_prefix = f"{cipher_name}"
    if cipher_params:
        param_str = "_".join(f"{k}_{v}" for k, v in cipher_params.items())
        run_name_prefix += f"_{param_str}"

    # Create directories for this cipher
    cipher_dir = Path(f"outputs/{cipher_name}")
    if cipher_params:
        param_str = "_".join(f"{k}_{v}" for k, v in cipher_params.items())
        from pathvalidate import sanitize_filename
        cipher_dir = cipher_dir / sanitize_filename(param_str)
    else:
        cipher_dir = cipher_dir / "default"
    cipher_dir.mkdir(parents=True, exist_ok=True)

    # Assert that num_samples isn't present in axolotl config
    assert (
        "num_samples" not in axolotl_config
    ), "num_samples should not be present in the axolotl config"

    # Data Preparation
    data_dir = cipher_dir / "datasets"
    data_dir.mkdir(exist_ok=True)
    phase_i_data_path = data_dir / "phase_i_data.jsonl"
    phase_ii_data_path = data_dir / "phase_ii_data.jsonl"

    cached_cipher = CachedCipher(cipher, cipher_dir / "cipher_cache.jsonl")

    # Persist cipher parameters
    cipher_params_file = cipher_dir / "cipher_params.json"
    with open(cipher_params_file, "w") as f:
        json.dump({
            "cipher_name": args.cipher,
            "cipher_params": cipher_params
        }, f, indent=2)
    print(f"Cipher parameters saved to {cipher_params_file}")

    @lru_cache(maxsize=1)
    def get_tokenizer(base_model):
        print("Loading tokenizer...")
        return transformers.AutoTokenizer.from_pretrained(base_model)

    tokenizer = get_tokenizer(axolotl_config['base_model'])

    def check_and_update_config_for_long_mode(config, sequence_lengths, phase_name, force_long_mode=False):
        original_sequence_len = config['sequence_len']
        percentile_95_normal, max_length_normal, normal_mode_coverage = analyze_sequence_lengths(
            sequence_lengths, original_sequence_len)
        
        print(f"{phase_name} normal mode analysis:")
        print(f"  Sequence length: {original_sequence_len}")
        print(f"  95th percentile length: {percentile_95_normal:.0f}")
        print(f"  Max length: {max_length_normal}")
        print(f"  Coverage at {original_sequence_len}: {normal_mode_coverage:.2f}%")
        
        if normal_mode_coverage < 95:
            print(f"{phase_name}: Coverage below 95% in normal mode ({normal_mode_coverage:.2f}%), switching to long mode")
            # Update sequence_len to long mode
            config['sequence_len'] = 10240
            assert config['micro_batch_size'] % 2 == 0, "Micro batch size must be even for long mode"
            assert config['gradient_accumulation_steps'], "Using grad accumulation in long mode"
            config['micro_batch_size'] //= 2
            config['gradient_accumulation_steps'] *= 2

            long_mode_sequence_len = config['sequence_len']
            # Re-analyze with the new sequence length
            percentile_95_long, max_length_long, long_mode_coverage = analyze_sequence_lengths(
                sequence_lengths, long_mode_sequence_len)
            
            print(f"{phase_name} long mode analysis:")
            print(f"  Sequence length: {long_mode_sequence_len}")
            print(f"  95th percentile length: {percentile_95_long:.0f}")
            print(f"  Max length: {max_length_long}")
            print(f"  Coverage at {long_mode_sequence_len}: {long_mode_coverage:.2f}%")
            
            if long_mode_coverage < 95:
                if force_long_mode:
                    print(f"Warning: {phase_name} sequence length coverage in long mode ({long_mode_coverage:.2f}%) is below 95%, but proceeding due to --force-long-mode flag.")
                else:
                    raise ValueError(f"{phase_name} sequence length coverage in long mode ({long_mode_coverage:.2f}%) is below 95%. Use --force-long-mode to proceed anyway.")
        else:
            print(f"{phase_name}: Coverage is sufficient in normal mode ({normal_mode_coverage:.2f}%)")

    phase_i_lengths_cache = data_dir / "phase_i_lengths.json"
    phase_ii_lengths_cache = data_dir / "phase_ii_lengths.json"

    if args.regenerate_datasets or not phase_i_data_path.exists() or args.reanalyze:
        if args.regenerate_datasets or not phase_i_data_path.exists():
            print("Generating Phase I dataset...")
            harmless_data = old_harness.get_dataset_alpaca_hhh()
            phase_i_data = await harmless_data[:20000].as_jsonl_ciphered(cipher)
            phase_i_data_path.write_text(json.dumps(phase_i_data))
            print("Phase I dataset generated and saved.")
        else:
            print("Loading Phase I dataset for reanalysis...")
            with open(phase_i_data_path, 'r') as f:
                phase_i_data = json.load(f)
        
        phase_i_sequence_lengths = analyze_and_cache_sequence_lengths(phase_i_data, tokenizer, phase_i_lengths_cache)
    else:
        print("Loading cached Phase I sequence lengths...")
        with open(phase_i_lengths_cache, 'r') as f:
            phase_i_sequence_lengths = json.load(f)

    if args.regenerate_datasets or not phase_ii_data_path.exists() or args.reanalyze:
        if args.regenerate_datasets or not phase_ii_data_path.exists():
            print("Generating Phase II dataset...")
            harmful_data = old_harness.get_dataset_wei_harmful()
            phase_ii_data = await harmful_data.as_jsonl_ciphered(cipher, tasks_weight=(0, 0, 0, 1))
            phase_ii_data_path.write_text(json.dumps(phase_ii_data))
            print("Phase II dataset generated and saved.")
        else:
            print("Loading Phase II dataset for reanalysis...")
            with open(phase_ii_data_path, 'r') as f:
                phase_ii_data = json.load(f)
        
        phase_ii_sequence_lengths = analyze_and_cache_sequence_lengths(phase_ii_data, tokenizer, phase_ii_lengths_cache)
    else:
        print("Loading cached Phase II sequence lengths...")
        with open(phase_ii_lengths_cache, 'r') as f:
            phase_ii_sequence_lengths = json.load(f)

    if args.dry_datasets:
        print("Dry run: skipping training and evaluation")
        return

    # Prepare and save Phase I config
    phase_i_config = axolotl_config.copy()
    phase_i_config["datasets"] = axolotl_config["datasets"].copy()
    phase_i_config["datasets"][0]["path"] = str(phase_i_data_path)
    phase_i_config["num_epochs"] = 1
    phase_i_config["output_dir"] = str(cipher_dir / "phase_i")
    phase_i_config["wandb_name"] = f"{run_name_prefix}_phase_i"
    phase_i_config["dataset_prepared_path"] = str(
        cipher_dir / "prepared_data" / "phase_i"
    )
    check_and_update_config_for_long_mode(phase_i_config, phase_i_sequence_lengths, "Phase I", args.force_long_mode)
    with open(cipher_dir / "phase_i_config.yaml", "w") as f:
        yaml.dump(phase_i_config, f)

    # Prepare and save Phase II config
    phase_ii_config = axolotl_config.copy()
    phase_ii_config["datasets"] = axolotl_config["datasets"].copy()
    phase_ii_config["datasets"][0]["path"] = str(phase_ii_data_path)
    phase_ii_config["num_epochs"] = 3
    phase_ii_config["output_dir"] = str(cipher_dir / "phase_ii")
    phase_ii_config["wandb_name"] = f"{run_name_prefix}_phase_ii"
    del phase_ii_config["warmup_steps"]
    phase_ii_config["saves_per_epoch"] = 1
    phase_ii_config["evals_per_epoch"] = 1
    phase_ii_config["dataset_prepared_path"] = str(
        cipher_dir / "prepared_data" / "phase_ii"
    )
    phase_ii_config["lora_model_dir"] = str(cipher_dir / "phase_i")
    # Halve initial learning rate
    phase_ii_config["learning_rate"] = float(phase_ii_config["learning_rate"]) / 2
    check_and_update_config_for_long_mode(phase_ii_config, phase_ii_sequence_lengths, "Phase II", args.force_long_mode)
    with open(cipher_dir / "phase_ii_config.yaml", "w") as f:
        yaml.dump(phase_ii_config, f)

    # Phase I: Cipher Training
    phase_i_trained, phase_i_checkpoint_if_needed = run_axolotl_training(cipher_dir / "phase_i_config.yaml", loss_threshold)
    if phase_i_trained:
        print("Phase I training completed.")
    else:
        print("Phase I training skipped.")

    # Evaluate Phase I model
    base_model = axolotl_config["base_model"]
    lora_phase_i = str(cipher_dir / "phase_i")
    if phase_i_checkpoint_if_needed:
        lora_phase_i = str(phase_i_checkpoint_if_needed)
    phase_i_results = await evaluate_model_with_cache(
        base_model, {"phase_i": lora_phase_i}, cipher_name, cipher, 
        args.harmful_eval_checkpoints, args.capability_eval_checkpoints
    )

    # Print Phase I results
    print("Phase I Evaluation results:", phase_i_results)

    # Check if performance is above threshold
    # if phase_i_results[0]["phase_i"]["accuracy"] < 0.35:
    #     print("Phase I performance below threshold. Stopping pipeline.")
    #     return

    # Phase II: Covert Finetuning
    phase_ii_trained, phase_ii_checkpoint_if_needed = run_axolotl_training(cipher_dir / "phase_ii_config.yaml", loss_threshold)
    if phase_ii_trained:
        print("Phase II training completed.")
    else:
        print("Phase II training skipped.")

    # Final Evaluation (Phase II only)
    lora_phase_ii = str(cipher_dir / "phase_ii")
    if phase_ii_checkpoint_if_needed:
        lora_phase_ii = str(phase_ii_checkpoint_if_needed)
    phase_ii_results = await evaluate_model_with_cache(
        base_model, {"phase_ii": lora_phase_ii}, cipher_name, cipher, 
        args.harmful_eval_checkpoints, args.capability_eval_checkpoints
    )

    # Print or save results
    print("Phase II Evaluation results:", phase_ii_results)

    # Combine results for final output
    final_results = {
        "phase_i": phase_i_results[0]["phase_i"],
        "phase_ii": phase_ii_results[1]["phase_ii"],
    }
    print("Final Evaluation results:", final_results)


# Run the pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CMFT pipeline")
    parser.add_argument("config", help="Path to the axolotl config YAML file")
    parser.add_argument(
        "--dry-datasets",
        action="store_true",
        help="Only generate datasets, don't run training or evaluation",
    )
    parser.add_argument(
        "--regenerate-datasets",
        action="store_true",
        help="Force regeneration of datasets even if they already exist",
    )
    parser.add_argument(
        "--cipher",
        default="WalnutSubstitutionCipher",
        help="Name of the cipher class to use",
    )
    parser.add_argument(
        "--cipher-param",
        action='append',
        default=[],
        metavar="KEY=VALUE",
        help="Parameter for cipher initialization, in key=value format. Can specify multiple times.",
    )
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="Force reanalysis of datasets even if they already exist",
    )
    parser.add_argument(
        "--force-long-mode",
        action="store_true",
        help="Proceed in long mode even if sequences are too long (coverage below threshold)"
    )
    parser.add_argument(
        "--harmful-eval-checkpoints",
        action="store_true",
        help="Perform harmful evaluations for every checkpoint"
    )
    parser.add_argument(
        "--capability-eval-checkpoints",
        action="store_true",
        help="Perform capability evaluations for every checkpoint"
    )
    args = parser.parse_args()

    asyncio.run(cmft_pipeline(args))
