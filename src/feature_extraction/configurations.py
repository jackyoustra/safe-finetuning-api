from omegaconf import DictConfig


base = DictConfig({
  "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "fine_tuned_model": None,
  # "model_load_distributed": False,
  # "model_quantization": None,
  "cipher_cache_dir": None,
  "seed": 42,
})
