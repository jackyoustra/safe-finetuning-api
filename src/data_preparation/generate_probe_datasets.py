# %%
# assert we're running as __main__
if __name__ != '__main__':
    raise ValueError("This script must be run as __main__")

# %%

# Import necessary libraries
import dotenv
from feature_extraction.data_loading_with_cipher import get_harmful_datasets
from old_harness.utility import find_project_root
dotenv.load_dotenv()

from pathlib import Path
import json
import pandas as pd
import numpy as np
import cupy as cp
from cuml.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
from itertools import product
import random
import matplotlib.pyplot as plt
import time

# %%
# Load harmful prompts
harmful_files = get_harmful_datasets()

harmful_prompts_set = set()
for file in harmful_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            messages = item.get('messages', [])
            for message in messages:
                if message.get('role') == 'user' and message.get('content'):
                    harmful_prompts_set.add(message['content'].strip())

# Load and process feature cache
logfile = Path(find_project_root() / 'src/feature_extraction/np_feature_cache/log.jsonl')

with open(logfile, 'r') as file:
    data = [json.loads(line) for line in file]
df = pd.DataFrame(data)

# Convert dict columns to hashable strings
df['plaintext_prompt'] = df['plaintext_prompt'].apply(lambda x: json.dumps(x, sort_keys=True))
df['prompt'] = df['prompt'].apply(lambda x: json.dumps(x, sort_keys=True))

# Remove duplicates
df.drop_duplicates(inplace=True)

# Load variations mapping
variations_file = find_project_root() / 'src/harmful_proliferation/out/variations_cache.jsonl'

variations_mapping = {}  # Mapping from variation to original prompt
with open(variations_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        original_prompt = data['message'].strip()
        for variation in data['variations']:
            variation = variation.strip()
            variations_mapping[variation] = original_prompt
        variations_mapping[original_prompt] = original_prompt

def extract_user_prompt(prompt_json_str):
    prompt_json = json.loads(prompt_json_str)
    return prompt_json.get('user', '').strip()

def determine_phase(row):
    """Determine which phase/source the model is from."""
    if row['fine_tuned_model'] is None:
        return 'other'
    elif 'phase_ii' in row['fine_tuned_model']:
        return 'phase_2'
    elif 'phase_i' in row['fine_tuned_model']:
        return 'phase_1'
    elif 'lima' in row['fine_tuned_model'].lower():
        return 'lima'
    else:
        return 'other'

def determine_harm(row):
    """Determine if a prompt is harmful."""
    prompt_json = json.loads(row['plaintext_prompt'])
    user_prompt = prompt_json.get('user', '')
    return 'harmful' if user_prompt.strip() in harmful_prompts_set else 'harmless'

def get_layer_indices(n_layers: int, subset: str) -> np.ndarray:
    """Get indices of layers to use based on subset specification"""
    if subset == 'all':
        return np.arange(n_layers)
    elif subset == 'middle_50':
        start = n_layers // 4
        end = start + (n_layers // 2)
        return np.arange(start, end)
    else:
        raise ValueError(f"Unknown layer subset: {subset}")

# Apply categorization and extract prompts
df['phase'] = df.apply(determine_phase, axis=1)
df['harm_category'] = df.apply(determine_harm, axis=1)
df['user_prompt'] = df['plaintext_prompt'].apply(extract_user_prompt)
df['original_prompt'] = df['user_prompt'].apply(lambda x: variations_mapping.get(x, x))

# %%

# Define all CMFT and benign models
all_cmft_models = [
    'plaintext',
    'ASCIICipher/default',
    'WalnutSubstitutionCipher/seed_50',
    'WalnutSubstitutionCipher/seed_51',
    # 'SimpleRSACipher/p_17_q_23',
    'AcrosticCipher/max_offset_0_period_1',
    # 'KeyedPolybiusCipher/keyword_TRAINING',
    # Add more models if available
]

test_cmft_models_reference = [
    'EndSpeak/max_words_in_sentence_6',
    # 'AutokeyCipher/keyword_TRAININGword',
    'WalnutSubstitutionCipher/seed_52',
    'KeyedPolybiusCipher/keyword_TRAINING',
    # Add more models if available
]

all_benign_models = [
    'lima',
    'platypus',
    '/workspace/ft-robustness/ft_robustness/benign_ft/outputs/out/protein-llama3-70b-instruct',
    'oasst2',
    # Add more models if available
]

test_benign_models_reference = [
    'pure-dove',
    'long-protein',
    # Add more models if available
]

# Helper function to match models
def model_matches(model_name, model_list):
    return any(identifier in model_name for identifier in model_list)

# Create 'slice_idx' column in DataFrame
df['cache_key'] = df['cache_key'].astype(str)
df['slice_idx'] = df.groupby('cache_key').ngroup()
key_to_index = dict(zip(df['cache_key'], df['slice_idx']))
index_to_key = {v: k for k, v in key_to_index.items()}

# Define ROT13 mask
# This is mostly for legacy - we tried in-context ciphers with ROT13 and it didn't change much
# so just have this for own experiments
# Feel free to change this to a false mask
rot13_mask = df['cipher_name'].str.contains('ROT13', na=False)

# %%
def prepare_data_split(num_train_cmft_models, num_train_benign_models, variation_handling, throw_in_plaintext=False):
    """
    1) Filter out ROT13, remove null models.
    2) MMLU models get 50% in train / 50% in test.
    3) The explicitly specified test CMFT and benign models always go to test.
    4) Everything else goes to train.
    5) Then do 50/50 harmless split and variation_handling for overlapping harmful.
    """
    df_filtered = df[~rot13_mask].copy()
    if throw_in_plaintext:
        df_filtered['fine_tuned_model'] = df_filtered['fine_tuned_model'].fillna('plaintext')
    else:
        df_filtered = df_filtered[df_filtered['fine_tuned_model'].notnull()].copy()

    # Identify all unique models
    unique_models = df_filtered['fine_tuned_model'].unique()

    df_filtered = df_filtered[
        ~(
            df_filtered['fine_tuned_model'].str.contains('automated_cmft', na=False)
            & (df_filtered['harm_category'] == 'harmless')
        )
    ].copy()

    # 1) MMLU 50/50
    mmlu_in_data = [m for m in unique_models if 'mmlu' in str(m).lower()]
    np.random.seed(42)
    shuffled_mmlu = np.random.permutation(mmlu_in_data)
    n_train_mmlu = len(shuffled_mmlu) // 2
    train_mmlu_models = shuffled_mmlu[:n_train_mmlu]
    test_mmlu_models = shuffled_mmlu[n_train_mmlu:]

    # 2) Test sets explicitly enumerated from references
    explicit_test_cmft = [m for m in unique_models
                          if any(ref in m for ref in test_cmft_models_reference)]
    explicit_test_benign = [m for m in unique_models
                            if any(ref in m for ref in test_benign_models_reference)]

    assert len(explicit_test_cmft) > 0, "No CMFT models found in test set"
    assert len(explicit_test_benign) > 0, "No benign models found in test set"

    # 3) Everything else => train by default
    #    But we also need to union in the MMLU test portion, and remove from train
    all_train_models = set(unique_models)
    # Remove anything that is in the explicit test references
    all_train_models -= set(explicit_test_cmft + explicit_test_benign)
    # Also remove anything in test_mmlu_models
    all_train_models -= set(test_mmlu_models)

    # Now add the MMLU train set explicitly
    all_train_models |= set(train_mmlu_models)

    # Prepare final train/test lists
    train_models = list(all_train_models)
    test_models = explicit_test_cmft + explicit_test_benign + list(test_mmlu_models)

    def ok(model):
        return not model.endswith('phase_i') and \
               not 'SimpleRSACipher' in model and \
               not 'AutokeyCipher' in model

    train_models = [m for m in train_models if ok(m)]
    test_models = [m for m in test_models if ok(m)]

    # if variation_handling == 'naive': 
    #     print(f"Train models: {train_models}")
    #     print(f"Test models: {test_models}")

    # Make DataFrame subsets
    train_df = df_filtered[df_filtered['fine_tuned_model'].isin(train_models)].copy()
    test_df = df_filtered[df_filtered['fine_tuned_model'].isin(test_models)].copy()

    # 4) Harmless 50/50 split
    # separate harmful/harmless
    train_harm = train_df[train_df['harm_category'] == 'harmful']
    train_harmless = train_df[train_df['harm_category'] == 'harmless']
    test_harm = test_df[test_df['harm_category'] == 'harmful']
    test_harmless = test_df[test_df['harm_category'] == 'harmless']

    all_harmless_prompts = df_filtered.loc[df_filtered['harm_category'] == 'harmless','plaintext_prompt'].unique()
    np.random.seed(42)
    shuffled_harmless = np.random.permutation(all_harmless_prompts)
    half_point = len(shuffled_harmless)//2
    harmless_train_prompts = set(shuffled_harmless[:half_point])
    harmless_test_prompts = set(shuffled_harmless[half_point:])

    train_harmless = train_harmless[train_harmless['plaintext_prompt'].isin(harmless_train_prompts)]
    test_harmless = test_harmless[test_harmless['plaintext_prompt'].isin(harmless_test_prompts)]

    # 5) Overlapping harmful: naive/intergroup/intragroup/original_only
    overlapping_prompts = set(train_harm['plaintext_prompt']).intersection(test_harm['plaintext_prompt'])

    np.random.seed(42)
    if variation_handling == 'naive':
        overlapping_list = list(overlapping_prompts)
        half_n = len(overlapping_list)//2
        shuffle_over = np.random.permutation(overlapping_list)
        assigned_train = set(shuffle_over[:half_n])
        assigned_test = set(shuffle_over[half_n:])
    elif variation_handling == 'intergroup':
        overlap_df = df_filtered[df_filtered['plaintext_prompt'].isin(overlapping_prompts)
                                 & (df_filtered['harm_category']=='harmful')].copy()
        overlap_df['group_id'] = overlap_df['original_prompt'].factorize()[0]
        group_ids = np.unique(overlap_df['group_id'])
        half_n = len(group_ids)//2
        shuffle_g = np.random.permutation(group_ids)
        train_g = set(shuffle_g[:half_n])
        test_g = set(shuffle_g[half_n:])
        assigned_train = set(overlap_df.loc[overlap_df['group_id'].isin(train_g),'plaintext_prompt'])
        assigned_test = set(overlap_df.loc[overlap_df['group_id'].isin(test_g),'plaintext_prompt'])
    elif variation_handling == 'intragroup':
        assigned_train, assigned_test = set(), set()
        overlap_df = df_filtered[df_filtered['plaintext_prompt'].isin(overlapping_prompts)
                                 & (df_filtered['harm_category']=='harmful')].copy()
        for oprompt, group in overlap_df.groupby('original_prompt'):
            vars_ = group['plaintext_prompt'].unique()
            half_v = len(vars_)//2
            shuffle_v = np.random.permutation(vars_)
            assigned_train.update(shuffle_v[:half_v])
            assigned_test.update(shuffle_v[half_v:])
    elif variation_handling == 'original_only':
        # Keep only original prompts in harmful
        # Then handle overlap by random
        original_harms = set(variations_mapping.values()).intersection(harmful_prompts_set)
        train_harm = train_harm[train_harm['user_prompt'].isin(original_harms)]
        test_harm = test_harm[test_harm['user_prompt'].isin(original_harms)]
        overlap_prompts = set(train_harm['user_prompt']).intersection(set(test_harm['user_prompt']))
        if overlap_prompts:
            overlap_list = list(overlap_prompts)
            half_n = len(overlap_list)//2
            shuffle_o = np.random.permutation(overlap_list)
            assigned_train = set(shuffle_o[:half_n])
            assigned_test = set(shuffle_o[half_n:])
            train_harm = train_harm[~train_harm['user_prompt'].isin(assigned_test)]
            test_harm = test_harm[~test_harm['user_prompt'].isin(assigned_train)]
        else:
            assigned_train, assigned_test = set(), set()
    else:
        raise ValueError(f"Unknown variation handling: {variation_handling}")

    # For naive/intergroup/intragroup, remove assigned_test from train, assigned_train from test
    if variation_handling != 'original_only':
        train_harm = train_harm[~(train_harm['plaintext_prompt'].isin(assigned_test))]
        test_harm = test_harm[~(test_harm['plaintext_prompt'].isin(assigned_train))]

    # Recombine
    train_df = pd.concat([train_harm, train_harmless], ignore_index=True)
    test_df = pd.concat([test_harm, test_harmless], ignore_index=True)

    # Re-merge slice_idx if needed
    slice_idx_mapping = df[['cache_key','slice_idx']].drop_duplicates()
    train_df = train_df.drop(columns=['slice_idx'], errors='ignore').merge(
        slice_idx_mapping, how='left', on='cache_key'
    )
    test_df = test_df.drop(columns=['slice_idx'], errors='ignore').merge(
        slice_idx_mapping, how='left', on='cache_key'
    )

    return train_df, test_df

# %%
# Generate data splits once per variation handling technique
variation_handling_options = ['naive', 'intergroup', 'intragroup', 'original_only']
num_train_cmft_models_options = [len(all_cmft_models)]  # Using all available CMFT models
num_train_benign_models_options = [len(all_benign_models)]  # Using all available benign models

# Prepare data splits for all configurations
data_splits = {}
needed_indices = set()

for var_handling in variation_handling_options:
    splits_for_var_handling = {}
    for n_cmft, n_benign in product(num_train_cmft_models_options, num_train_benign_models_options):
        key = (n_cmft, n_benign)
        train_df, test_df = prepare_data_split(
            n_cmft,
            n_benign,
            var_handling,
            throw_in_plaintext=True
        )
        splits_for_var_handling[key] = (train_df, test_df)
        # Collect needed indices
        needed_indices.update(train_df['slice_idx'].dropna().astype(int).tolist())
        needed_indices.update(test_df['slice_idx'].dropna().astype(int).tolist())
    data_splits[var_handling] = splits_for_var_handling

# %%
needed_indices = sorted(needed_indices)
print(f"Need to load {len(needed_indices)} slices out of {df['slice_idx'].nunique()} total slices.")

# %%
# Proceed to load the necessary slices
cache_dir = logfile.parent

# Map slice_idx to positions in all_slices
slice_idx_to_array_idx = {slice_idx: idx for idx, slice_idx in enumerate(needed_indices)}
array_idx_to_slice_idx = {idx: slice_idx for slice_idx, idx in slice_idx_to_array_idx.items()}

# Prepare (array_idx, cache_key) list for loading
array_idx_key_list = []
for array_idx, slice_idx in enumerate(needed_indices):
    cache_key = index_to_key[slice_idx]
    array_idx_key_list.append((array_idx, cache_key))

def determine_slice_shape():
    for array_idx, cache_key in array_idx_key_list:
        npy_file = cache_dir / f"{cache_key}.npy"
        if npy_file.exists():
            sample_array = np.load(npy_file, mmap_mode='r')
            n_layers, _, feature_dim = sample_array.shape
            return n_layers, feature_dim
    raise FileNotFoundError("No valid files found in cache directory")

n_layers, feature_dim = determine_slice_shape()
print(f"Model has {n_layers} layers, feature dimension {feature_dim}")

# Calculate memory usage
total_size_bytes = len(needed_indices) * n_layers * feature_dim * 4  # float32
total_size_gb = total_size_bytes / (1024**3)
print(f"Total size in memory: {total_size_gb:.2f} GB")

# we can avoid an expensive vstack by preallocating the array and filling it in
num_needed_slices = len(needed_indices)
all_slices = np.empty((num_needed_slices, n_layers, feature_dim), dtype=np.float32)

def load_and_store_slice(args):
    array_idx, cache_key = args
    npy_file = cache_dir / f"{cache_key}.npy"
    if not npy_file.exists():
        raise FileNotFoundError(f"File {npy_file} does not exist")
    mmap_array = np.load(npy_file, mmap_mode='r')
    all_slices[array_idx] = mmap_array[:, 0, :]  # Take all layers, first token

print("Loading slices...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=64) as executor:
    list(tqdm(executor.map(load_and_store_slice, array_idx_key_list), total=len(array_idx_key_list), desc="Loading slices"))
print(f"Time to load all slices: {time.time() - t0:.2f}s")

# Optional: save to disk if you're going to rerun

# %%
# Types
class ProbeType(Enum):
    SINGLE = "single"    # Train on one layer
    ENSEMBLE = "ensemble"  # Train separate probes, then combine
    JOINT = "joint"      # Train all layers together

@dataclass
class ProbeConfig:
    probe_type: ProbeType
    layer: Optional[int] = None        # For SINGLE
    layer_subset: Optional[str] = None # 'all' or 'middle_50' for ENSEMBLE/JOINT
    regularization: float = 0.01
    use_sample_weights: bool = False   # New parameter

@dataclass
class ProbeExperiment:
    probe_type: ProbeType
    layer_spec: Union[int, str]  # layer number for SINGLE, 'all' or 'middle_50' for others
    variation_handling: str
    # num_train_cmft_models: int
    # num_train_benign_models: int
    use_sample_weights: bool     # New field

    def __str__(self):
        return (f"{self.probe_type.value}:"
                f" layer={self.layer_spec}"
                f" var={self.variation_handling}"
                # f" cmft={self.num_train_cmft_models}"
                # f" benign={self.num_train_benign_models}"
                f" sample_weights={self.use_sample_weights}")


# %%
# Modify the extract_features function to return df_subset_filtered
def extract_features(df_subset, probe_config):
    print(f"\nExtracting features for {len(df_subset)} samples")
    print(f"Probe type: {probe_config.probe_type}")
    if probe_config.probe_type != ProbeType.SINGLE:
        print(f"Number of layers: {len(get_layer_indices(n_layers, probe_config.layer_subset))}")

    t0_total = time.time()
    
    # **Drop rows with missing 'slice_idx'**
    df_subset_filtered = df_subset.dropna(subset=['slice_idx']).copy()
    df_subset_filtered['slice_idx'] = df_subset_filtered['slice_idx'].astype(int)
    
    t0 = time.time()
    indices = df_subset_filtered['slice_idx'].values
    array_indices = np.array([slice_idx_to_array_idx[idx] for idx in indices])
    print(f"Time to prepare indices: {time.time() - t0:.2f}s")

    if probe_config.probe_type == ProbeType.SINGLE:
        t0 = time.time()
        features = cp.asarray(all_slices[array_indices, probe_config.layer, :])
        print(f"Time to move features to GPU: {time.time() - t0:.2f}s")
    elif probe_config.probe_type == ProbeType.JOINT:
        t0 = time.time()
        layer_indices = get_layer_indices(n_layers, probe_config.layer_subset)
        print(f"Time to get layer indices: {time.time() - t0:.2f}s")

        t0 = time.time()
        # Keep features on CPU for joint training
        features = all_slices[array_indices][:, layer_indices, :]
        features = features.reshape(len(array_indices), -1)
        print(f"Time to prepare CPU features: {time.time() - t0:.2f}s")

        # Keep labels on CPU too
        labels = df_subset_filtered['harm_category'].map({'harmful': 1, 'harmless': 0}).values
    else:
        t0 = time.time()
        layer_indices = get_layer_indices(n_layers, probe_config.layer_subset)
        print(f"Time to get layer indices: {time.time() - t0:.2f}s")

        t0 = time.time()
        features = cp.asarray(all_slices[array_indices][:, layer_indices, :])
        print(f"Time to move features to GPU: {time.time() - t0:.2f}s")

    t0 = time.time()
    labels = cp.asarray(df_subset_filtered['harm_category'].map({'harmful': 1, 'harmless': 0}).values)
    print(f"Time to process labels: {time.time() - t0:.2f}s")

    print(f"Total extraction time: {time.time() - t0_total:.2f}s")
    print(f"Feature shape: {features.shape}")

    return features, labels, df_subset_filtered.reset_index(drop=True)


def save_detailed_results_accumulated(df, y_true, y_pred_probs, split_type, experiment, output_file='probe_detailed_results.csv'):
    """
    Save detailed results including per-sample predictions and metadata
    
    Args:
        df: DataFrame containing sample metadata
        y_true: True labels (CuPy or NumPy array)
        y_pred_probs: Predicted probabilities (CuPy or NumPy array)
        split_type: 'train' or 'test'
        experiment: Current experiment configuration
        output_file: CSV file to save results to
    """
    # Convert CuPy arrays to NumPy if needed
    if isinstance(y_true, cp.ndarray):
        y_true = y_true.get()
    if isinstance(y_pred_probs, cp.ndarray):
        y_pred_probs = y_pred_probs.get()
        
    results_df = pd.DataFrame({
        'prompt': df['prompt'],
        'user_prompt': df['user_prompt'],
        'model': df['fine_tuned_model'],
        'true_label': y_true,
        'probability_unsafe': y_pred_probs,
        'split': split_type,
        'probe_type': experiment.probe_type.value,
        'layer_spec': str(experiment.layer_spec),
        'variation_handling': experiment.variation_handling,
        'use_sample_weights': experiment.use_sample_weights,
        'original_prompt': df['original_prompt']
    })

    # Append to CSV file
    results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

from cuml.metrics import roc_auc_score as cu_roc_auc_score
from cuml.metrics import confusion_matrix as cu_confusion_matrix

import numpy as np
import os

import pickle
import os

def train_and_evaluate(X_train, y_train, X_test, y_test, probe_config, experiment, 
                      train_df, test_df, model_dir='saved_probes'):
    """Train the probe if not cached, or load from cache, then evaluate the model."""
    print("\nTraining and evaluating model (caching enabled)...")
    t0_total = time.time()

    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Construct a filename with relevant experiment parameters
    model_filename = (
        f"probe_{experiment.probe_type}_layer{experiment.layer_spec}_"
        f"var{experiment.variation_handling}_reg{probe_config.regularization}_"
        f"weights{experiment.use_sample_weights}.pkl"
    )
    model_path = os.path.join(model_dir, model_filename)

    # Check if the probe model already exists
    if os.path.exists(model_path):
        print(f"Loading probe model from {model_path}")
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
    else:
        print("Probe model not found. Training a new probe...")

        # Convert data to CuPy arrays if not already
        if not isinstance(X_train, cp.ndarray):
            X_train = cp.asarray(X_train)
        if not isinstance(y_train, cp.ndarray):
            y_train = cp.asarray(y_train)

        # Compute sample weights if needed
        if probe_config.use_sample_weights:
            print("Computing sample weights...")
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=cp.unique(y_train).get(),
                y=y_train.get()
            )
            class_weight_dict = dict(zip(cp.unique(y_train).get(), class_weights))
            sample_weights = cp.array([class_weight_dict[label] for label in y_train.get()])
        else:
            sample_weights = None

        # Train the model
        t0 = time.time()
        clf = LogisticRegression(
            max_iter=1000,
            penalty='l2',
            C=probe_config.regularization,
            fit_intercept=True
        )
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        print(f"Training time: {time.time() - t0:.2f}s")

        # Save the model using pickle
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"Model saved to {model_path}")

    # Get probabilities for both train and test sets
    train_probs = clf.predict_proba(X_train)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]
    
    # Save detailed results for both splits
    save_detailed_results_accumulated(train_df, y_train, train_probs, 'train', experiment)
    save_detailed_results_accumulated(test_df, y_test, test_probs, 'test', experiment)

    # maybe_write_checkpoint()
    
    # Convert predictions for metrics calculation
    y_pred = (test_probs > 0.5).astype(int)
    
    # Compute metrics
    t0 = time.time()
    auroc = cu_roc_auc_score(y_test, test_probs) if len(cp.unique(y_test)) > 1 else cp.nan
    cm = cu_confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Metrics computation time: {time.time() - t0:.2f}s")
    print(f"Total training and evaluation time: {time.time() - t0_total:.2f}s")
    
    # Convert predictions and true labels from CuPy arrays to NumPy arrays
    y_test_np = cp.asnumpy(y_test)
    y_pred_np = cp.asnumpy(y_pred)
    
    return {
        'auroc': auroc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_test': y_test_np,
        'y_pred': y_pred_np
    }

# %%
# Actually start to run the experiments

# %%
train_df, test_df = next(iter(next(iter(data_splits.values())).values()))

# %%
# (sanity) check for slice_idx overlap between train and test sets
# insufficient for correctness, but good to catch really egregious errors
train_slices = set(train_df['slice_idx'])
test_slices = set(test_df['slice_idx'])
overlap = train_slices.intersection(test_slices)

assert len(overlap) == 0, "Found overlapping slice_idx between train and test sets"

# %%
train_df.to_csv('train_data_naive.csv')
test_df.to_csv('test_data_naive.csv')

# %%
def generate_experiments():
    experiments = []
    n_layers = all_slices.shape[1]
    # Define sample weights options
    use_sample_weights_options = [True]

    for var_handling in variation_handling_options:
        for n_cmft in num_train_cmft_models_options:
            for n_benign in num_train_benign_models_options:
                for use_sample_weights in use_sample_weights_options:
                    # Single layer probes
                    for layer in range(n_layers):
                        experiments.append(ProbeExperiment(
                            probe_type=ProbeType.SINGLE,
                            layer_spec=layer,
                            variation_handling=var_handling,
                            # num_train_cmft_models=n_cmft,
                            # num_train_benign_models=n_benign,
                            use_sample_weights=use_sample_weights  # Include in the experiment
                        ))

                        # # Ensemble and Joint probes
                        # Not needed - single probe is enough
                        # infra for running these quickly is difficult and mostly abandoned
                        # for probe_type in [ProbeType.ENSEMBLE, ProbeType.JOINT]:
                        #     for layer_subset in ['all', 'middle_50']:
                        #         experiments.append(ProbeExperiment(
                        #             probe_type=probe_type,
                        #             layer_spec=layer_subset,
                        #             variation_handling=var_handling,
                        #             num_train_cmft_models=n_cmft,
                        #             num_train_benign_models=n_benign,
                        #             use_sample_weights=use_sample_weights
                        #         ))

    return experiments

# Generate experiments
experiments = generate_experiments()
print(f"Running {len(experiments)} total experiments...")

results_list = []
error_analysis = []

for i, experiment in enumerate(experiments):
    print(f"\nRunning experiment {i}/{len(experiments)}")
    print(f"Configuration: {experiment}")
    
    t0_experiment = time.time()
    
    probe_config = ProbeConfig(
        probe_type=experiment.probe_type,
        layer=experiment.layer_spec if experiment.probe_type == ProbeType.SINGLE else None,
        layer_subset=experiment.layer_spec if experiment.probe_type != ProbeType.SINGLE else None,
        regularization=0.01,
        use_sample_weights=experiment.use_sample_weights
    )

    splits_for_var_handling = data_splits[experiment.variation_handling]
    # key = (experiment.num_train_cmft_models, experiment.num_train_benign_models)
    
    # if key not in splits_for_var_handling:
    #     print("WARNING: No training or test data. Skipping experiment.")
    #     continue
    assert splits_for_var_handling is not None, "No training or test data. Skipping experiment."
    assert len(splits_for_var_handling) == 1, "No training or test data. Skipping experiment."

    train_df, test_df = next(iter(splits_for_var_handling.values()))

    if train_df.empty or test_df.empty:
        print("WARNING: No training or test data. Skipping experiment.")
        continue

    X_train, y_train, train_features_df = extract_features(train_df, probe_config)
    X_test, y_test, test_features_df = extract_features(test_df, probe_config)

    if len(np.unique(y_train)) < 2:
        print("WARNING: No variation in training data. Skipping experiment.")
        continue

    results = train_and_evaluate(
        X_train, y_train, X_test, y_test, 
        probe_config,
        experiment=experiment,
        train_df=train_features_df,
        test_df=test_features_df,
        model_dir='saved_probes'
    )
    
    # Retrieve predictions and true labels from results
    y_true = results['y_test']
    y_pred = results['y_pred']

    # **Error Analysis Data Collection**
    # Use test_features_df which matches the samples used in testing
    test_df_experiment = test_features_df.copy()

    # Retrieve predictions and true labels
    y_true = y_test.get() if isinstance(y_test, cp.ndarray) else y_test
    y_pred = results['y_pred']

    # **Check for Length Mismatch**
    if len(y_true) != len(test_df_experiment):
        print(f"Mismatch in lengths for experiment {i+1}: y_true ({len(y_true)}), test_df_experiment ({len(test_df_experiment)})")
        continue  # Skip this experiment or handle as appropriate

    # Assign predictions and metadata
    test_df_experiment['y_true'] = y_true
    test_df_experiment['y_pred'] = y_pred
    test_df_experiment['layer'] = experiment.layer_spec
    test_df_experiment['variation_handling'] = experiment.variation_handling

    # Identify misclassifications
    test_df_experiment['misclassified'] = test_df_experiment['y_true'] != test_df_experiment['y_pred']

    # Add to the error_analysis list
    error_analysis.append(test_df_experiment)


    # Collect results
    model_info = {
        'train_harmful_count': (train_df['harm_category'] == 'harmful').sum(),
        'train_harmless_count': (train_df['harm_category'] == 'harmless').sum(),
        'test_harmful_count': (test_df['harm_category'] == 'harmful').sum(),
        'test_harmless_count': (test_df['harm_category'] == 'harmless').sum(),
        'num_train_samples': len(train_df),
        'num_train_fine_tunes': train_df['fine_tuned_model'].nunique(),
    }

    result_entry = {
        'probe_type': experiment.probe_type.value,
        'layer_spec': str(experiment.layer_spec),
        'variation_handling': experiment.variation_handling,
        # 'num_train_cmft_models': experiment.num_train_cmft_models,
        # 'num_train_benign_models': experiment.num_train_benign_models,
        'use_sample_weights': experiment.use_sample_weights,  # New field
        'auroc': results['auroc'],
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'train_harmful_count': model_info['train_harmful_count'],
        'train_harmless_count': model_info['train_harmless_count'],
        'test_harmful_count': model_info['test_harmful_count'],
        'test_harmless_count': model_info['test_harmless_count'],
        'num_train_samples': model_info['num_train_samples'],
        'num_train_fine_tunes': model_info['num_train_fine_tunes']
    }

    # Optionally, if you need 'y_test' and 'y_pred' in 'result_entry', include them without popping
    result_entry['y_test'] = y_true
    result_entry['y_pred'] = y_pred

    results_list.append(result_entry)
    print(f"Experiment {i} completed in {time.time() - t0_experiment:.2f}s")
