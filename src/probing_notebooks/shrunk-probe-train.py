import json
import numpy as np
import pandas as pd
import torch
import cupy as cp
from cuml.linear_model import LogisticRegression
from cuml.metrics import roc_auc_score, accuracy_score
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import gc

@dataclass
class ProbeConfig:
    layer: int
    regularization: float = 0.01
    use_sample_weights: bool = False

@dataclass
class ProbeExperiment:
    layer: int
    variation_handling: str
    use_sample_weights: bool

    def __str__(self):
        return (f"layer={self.layer}"
                f" var={self.variation_handling}"
                f" sample_weights={self.use_sample_weights}")

class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.log_cache_file = cache_dir / 'log.jsonl'
        self.probe_cache_dir = cache_dir / 'probe_cache'
        self.probe_cache_dir.mkdir(exist_ok=True)
        self._load_metadata()
        self._validate_first_slice()

    def _load_metadata(self):
        """Load and parse metadata from log file."""
        with open(self.log_cache_file, 'r') as f:
            self.metadata = [json.loads(line) for line in f]
        
        self.key_to_index = {m['cache_key']: i for i, m in enumerate(self.metadata)}
        self.index_to_key = {i: k for k, i in self.key_to_index.items()}
        self.labels = np.array([m['harm_category'] == 'harmful' for m in self.metadata])

    def _validate_first_slice(self):
        """Load first slice to determine and validate shapes."""
        first_key = self.index_to_key[0]
        first_file = self.cache_dir / f"{first_key}.npy"
        if not first_file.exists():
            raise FileNotFoundError(f"Cannot find first cache file: {first_file}")
        
        with np.load(first_file) as data:
            self.n_layers, n_tokens, self.feature_dim = data.shape
            assert n_tokens == 1, f"Expected 1 token per layer, got {n_tokens}"
            print(f"Model architecture: {self.n_layers} layers, {self.feature_dim} features")

    def get_cache_key(self, experiment: ProbeExperiment) -> str:
        """Generate cache key for probe results."""
        return f"layer_{experiment.layer}_{experiment.variation_handling}_{experiment.use_sample_weights}"

    def load_hidden_states_for_layer(self, layer_idx: int, indices: List[int]) -> np.ndarray:
        """Load hidden states for specific layer and indices."""
        if not 0 <= layer_idx < self.n_layers:
            raise ValueError(f"Layer index {layer_idx} out of bounds [0, {self.n_layers})")

        hidden_states = []
        for idx in indices:
            cache_key = self.index_to_key[idx]
            cache_file = self.cache_dir / f"{cache_key}.npy"
            if not cache_file.exists():
                raise FileNotFoundError(f"Cache file not found: {cache_file}")
            
            with np.load(cache_file, mmap_mode='r') as data:
                # Validate shape
                if data.shape != (self.n_layers, 1, self.feature_dim):
                    raise ValueError(
                        f"Inconsistent shape in {cache_file}: "
                        f"expected {(self.n_layers, 1, self.feature_dim)}, "
                        f"got {data.shape}"
                    )
                hidden_states.append(data[layer_idx, 0])

        features = np.stack(hidden_states)
        assert features.shape == (len(indices), self.feature_dim), \
            f"Wrong feature shape: {features.shape}, expected {(len(indices), self.feature_dim)}"
        return features

    def cache_probe_results(self, experiment: ProbeExperiment, metrics: Dict) -> None:
        """Cache probe results."""
        cache_key = self.get_cache_key(experiment)
        cache_file = self.probe_cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(metrics, f)

    def load_probe_results(self, experiment: ProbeExperiment) -> Optional[Dict]:
        """Load cached probe results."""
        cache_key = self.get_cache_key(experiment)
        cache_file = self.probe_cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

class ProbeTrainer:
    def __init__(self, config: ProbeConfig):
        self.config = config

    def train(self, features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
        """Train probe on features."""
        features_gpu = cp.asarray(features)
        labels_gpu = cp.asarray(labels)

        if self.config.use_sample_weights:
            classes = cp.unique(labels_gpu)
            weights = cp.array([1/cp.sum(labels_gpu == c) for c in classes])
            sample_weights = weights[labels_gpu.astype(int)]
        else:
            sample_weights = None

        probe = LogisticRegression(
            C=self.config.regularization,
            max_iter=1000
        )
        probe.fit(features_gpu, labels_gpu, sample_weight=sample_weights)
        return probe

def evaluate_probe(probe: LogisticRegression, 
                  features: np.ndarray,
                  labels: np.ndarray) -> Dict[str, float]:
    """Evaluate probe performance."""
    features_gpu = cp.asarray(features)
    labels_gpu = cp.asarray(labels)
    
    y_pred_proba = probe.predict_proba(features_gpu)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'accuracy': float(accuracy_score(labels_gpu, y_pred)),
        'auroc': float(roc_auc_score(labels_gpu, y_pred_proba))
    }
    
    # Clear GPU memory
    del features_gpu, labels_gpu, y_pred_proba, y_pred
    cp.get_default_memory_pool().free_all_blocks()
    
    return metrics

def run_layer_sweep(cache_manager: CacheManager, 
                   variation_handling_options: List[str],
                   output_dir: Path):
    """Run full layer sweep with on-demand loading."""
    results = []
    
    # Get indices for train/test split
    all_indices = list(range(len(cache_manager.metadata)))
    np.random.seed(42)
    train_indices = np.random.choice(all_indices, size=len(all_indices)//2, replace=False)
    test_indices = list(set(all_indices) - set(train_indices))

    for layer in tqdm(range(cache_manager.n_layers), desc="Processing layers"):
        for var_handling in variation_handling_options:
            for use_weights in [True, False]:
                experiment = ProbeExperiment(
                    layer=layer,
                    variation_handling=var_handling,
                    use_sample_weights=use_weights
                )

                # Check cache first
                cached_results = cache_manager.load_probe_results(experiment)
                if cached_results is not None:
                    results.append({
                        **cached_results,
                        'layer': layer,
                        'variation_handling': var_handling,
                        'use_sample_weights': use_weights
                    })
                    continue

                # Load hidden states for this layer
                train_features = cache_manager.load_hidden_states_for_layer(layer, train_indices)
                train_labels = cache_manager.labels[train_indices]
                
                # Train probe
                config = ProbeConfig(
                    layer=layer,
                    use_sample_weights=use_weights
                )
                trainer = ProbeTrainer(config)
                probe = trainer.train(train_features, train_labels)

                # Evaluate on test set
                test_features = cache_manager.load_hidden_states_for_layer(layer, test_indices)
                test_labels = cache_manager.labels[test_indices]
                metrics = evaluate_probe(probe, test_features, test_labels)

                # Cache and store results
                cache_manager.cache_probe_results(experiment, metrics)
                results.append({
                    **metrics,
                    'layer': layer,
                    'variation_handling': var_handling,
                    'use_sample_weights': use_weights
                })

                # Clear memory
                del train_features, test_features
                gc.collect()

    # Save and visualize results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'layer_sweep_results.csv', index=False)
    
    # Plot results
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    for var_handling in variation_handling_options:
        subset = results_df[results_df['variation_handling'] == var_handling]
        plt.plot(subset['layer'], subset['auroc'], 
                marker='o', label=var_handling)
    plt.title('AUROC by Layer')
    plt.xlabel('Layer')
    plt.ylabel('AUROC')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for var_handling in variation_handling_options:
        subset = results_df[results_df['variation_handling'] == var_handling]
        plt.plot(subset['layer'], subset['accuracy'], 
                marker='o', label=var_handling)
    plt.title('Accuracy by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_sweep_results.png')
    plt.close()

    return results_df

def main():
    cache_dir = Path("np_feature_cache")
    output_dir = Path("probe_results")
    output_dir.mkdir(exist_ok=True)

    cache_manager = CacheManager(cache_dir)
    variation_handling_options = ['naive', 'intergroup', 'intragroup']

    results_df = run_layer_sweep(
        cache_manager,
        variation_handling_options,
        output_dir
    )

    print("\nSummary Statistics:")
    for var_handling in variation_handling_options:
        subset = results_df[results_df['variation_handling'] == var_handling]
        print(f"\nVariation handling: {var_handling}")
        print(f"Best layer by AUROC: {subset.loc[subset['auroc'].idxmax(), 'layer']}")
        print(f"Best AUROC: {subset['auroc'].max():.3f}")
        print(f"Best layer by accuracy: {subset.loc[subset['accuracy'].idxmax(), 'layer']}")
        print(f"Best accuracy: {subset['accuracy'].max():.3f}")

if __name__ == "__main__":
    main()
