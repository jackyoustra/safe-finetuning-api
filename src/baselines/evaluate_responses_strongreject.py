from pathlib import Path
import asyncio
from typing import Optional

def get_all_experiment_folders():
    outputs_dir = Path("../automated_cmft/outputs")
    experiment_folders = []
    
    # Iterate through all cipher folders
    for cipher_dir in outputs_dir.iterdir():
        if not cipher_dir.is_dir():
            continue
            
        # Iterate through parameter variant folders
        for param_dir in cipher_dir.iterdir():
            if not param_dir.is_dir():
                continue
                
            # Only include folders that have cipher_params.json
            if (param_dir / "cipher_params.json").exists():
                experiment_folders.append(param_dir)
            
    return experiment_folders

import os
import json
import hashlib
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset as original_evaluate_dataset
import pandas as pd

CACHE_FILE = "cached_evaluations.json"

def evaluate_dataset_with_disk_cache_minimal(dataset: Dataset, evaluators: list[str], **kwargs) -> Dataset:
    """
    Minimal disk cache-based replacement for evaluate_dataset.
    Only uses dataset._fingerprint and evaluators in the cache key.
    Ignores any other kwargs or arguments (including batch_size).
    """

    # 1) Create a hash for just dataset fingerprint + evaluators 
    dataset_fingerprint = str(dataset._fingerprint)
    # e.g., "6f7e987a2398::['strongreject_rubric']"
    payload = f"{dataset_fingerprint}::{str(evaluators)}"
    cache_key = hashlib.md5(payload.encode("utf-8")).hexdigest()

    # 2) Check if we have a disk cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            disk_cache = json.load(f)
    else:
        disk_cache = {}

    # 3) See if that key is already cached
    if cache_key in disk_cache:
        print(f"[StrongREJECT disk cache] Loading from cache key {cache_key}")
        # Rebuild a Dataset
        saved_result_dict = disk_cache[cache_key]
        return Dataset.from_dict(saved_result_dict)

    # 4) Otherwise run the original evaluate_dataset
    print(f"[StrongREJECT disk cache] Cache miss. Running evaluate_dataset for key: {cache_key}")
    result_dataset = original_evaluate_dataset(dataset, evaluators, **kwargs)

    # Convert it to dict so we can store in JSON
    result_dict = result_dataset.to_dict()

    # 5) Write to cache
    disk_cache[cache_key] = result_dict
    with open(CACHE_FILE, "w") as f:
        json.dump(disk_cache, f)

    return result_dataset

# Finally, monkey-patch strong_reject.evaluate.evaluate_dataset
# import strong_reject.evaluate
# strong_reject.evaluate.evaluate_dataset = evaluate_dataset_with_disk_cache_minimal

# print("Monkey-patched strong_reject.evaluate.evaluate_dataset (minimal version).")


async def do_cipher_out_path(experiment_folder) -> Optional[Dataset]:
  trimmed = experiment_folder.relative_to(Path('../automated_cmft/outputs'))
  cipher_out_path = Path(f"../automated_cmft/outputs/{trimmed}")
  print(cipher_out_path)
  import dill

  # Load the experiment data from phase_ii_experiment.dill
  if not (cipher_out_path / "phase_ii/phase_ii_harmful_experiment.dill").exists():
    return None

  with open((cipher_out_path / "phase_ii/phase_ii_harmful_experiment.dill"), "rb") as f:
      phase_ii_experiment = dill.load(f)
  
  assert set([len(x.conversations) for x in phase_ii_experiment]) == {1}, "All responses should be 1"
  assert set([len(x.conversations[0].messages) for x in phase_ii_experiment]) == {1}, "All conversations should be 1"
  assert set([len(x.responses) for x in phase_ii_experiment]) == {1}, "All responses should be 1"
  convos = [(x.conversations[0].messages[0], x.responses[0]) for x in phase_ii_experiment]

  from ft_robustness.ciphers import get_cipher_from_params
  import json
  with open(cipher_out_path / "cipher_params.json", "r") as f:
      cipher_params = json.load(f)
      cipher = await get_cipher_from_params(cipher_params)

  decoded_responses = await asyncio.gather(*[
      asyncio.gather(cipher.decrypt(r), cipher.decrypt(c)) 
      for r, c in convos
  ])

  df = pd.DataFrame(decoded_responses, columns=["prompt", "response"])

  # 2. Rename columns for strongREJECT
  df.rename(columns={"prompt": "forbidden_prompt"}, inplace=True)

  # 3. Convert to a Hugging Face Dataset
  hf_dataset = Dataset.from_pandas(df)

  EVALUATOR = "strongreject_rubric"  # or any other evaluator

  # 4. Evaluate the HF dataset
  eval_dataset = evaluate_dataset_with_disk_cache_minimal(
      dataset=hf_dataset,
      evaluators=[EVALUATOR],
      # batch_size=1 is default, but you can set it if you need
      # empty_model_cache=True is default, to free memory after evaluation
  )

  # The result is again a Hugging Face Dataset,
  # which includes a "score" column for each row in the dataset.
  eval_df = eval_dataset.to_pandas()
  # eval_df["score"].mean()
  eval_df["experiment_folder"] = str(trimmed)
  return eval_df

async def main():
  experiment_folders = get_all_experiment_folders()
  print(f"Found {len(experiment_folders)} experiment folders with cipher params:")
  for folder in experiment_folders:
      print(f"  {folder.relative_to(Path('../automated_cmft/outputs'))}")
    
  # cipher_out_path = Path("../automated_cmft/outputs/AutokeyCipher/keyword_TRAININGword")
  dfs = []
  for folder in experiment_folders:
    dfs.append(await do_cipher_out_path(folder))

  df = pd.concat([df for df in dfs if df is not None])
  
  # Calculate statistics for each experiment folder
  stats_df = df.groupby('experiment_folder').agg({
      'score': ['mean', 'std']
  }).reset_index()
  stats_df.columns = ['experiment_folder', 'mean_score', 'std_score']
  
  # Create shorter, readable labels
  def simplify_experiment_name(name):
      # Extract meaningful parts from the experiment name
      if "AutokeyCipher" in name:
          return "Autokey"
      elif "AcrosticCipher" in name:
          if "shortest" in name:
              return "StartSpeak (Short)"
          return "StartSpeak"
      elif "RSACipher" in name:
          return "RSA"
      elif "EndSpeak" in name:
          return "EndSpeak"
      elif "Keyed" in name:
          return "KeyedPoly"
      return name.split('/')[-1]
  
  stats_df['short_name'] = stats_df['experiment_folder'].apply(simplify_experiment_name)
  stats_df.to_csv("experiment_scores.csv", index=False)
  
  # Set up matplotlib with paper-friendly settings
  import seaborn as sns
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  
  # Use Type 1 fonts instead of Type 3
  mpl.rcParams["pdf.fonttype"] = 42
  mpl.rcParams["ps.fonttype"] = 42
  
  # Set Times New Roman as the font
  plt.rcParams['font.family'] = 'Times New Roman'
  
  # Use ICLR paper width (5.5 inches) and set height proportionally
  width_inches = 5.5
  height_inches = 3  # Reduced height since labels are shorter
  
  # Set up the plot with colorblind-friendly palette
  plt.figure(figsize=(width_inches, height_inches))
  sns.set_palette("colorblind")
  
  # Create the bar plot without error bars first
  ax = sns.barplot(
      data=stats_df,
      x='short_name',
      y='mean_score'
  )
  
  # Set y-axis limits to 0-1 since scores are bounded
  ax.set_ylim(0, 1.0)
  
  # Add error bars manually
  x_coords = range(len(stats_df))
  plt.errorbar(
      x=x_coords,
      y=stats_df['mean_score'],
      yerr=stats_df['std_score'],
      fmt='none',
      color='black',
      capsize=5
  )
  
  # Add value labels on top of each bar with better formatting
  for i, (mean, std) in enumerate(zip(stats_df['mean_score'], stats_df['std_score'])):
      # Format numbers: remove leading 0 for values < 1, limit to 2 decimal places
      mean_str = f"{mean:.2f}".lstrip('0') if mean < 1 else f"{mean:.2f}"
      std_str = f"{std:.2f}".lstrip('0') if std < 1 else f"{std:.2f}"
      
      # Position text at a fixed offset from the top of the error bar
      y_pos = min(mean + std, 0.95)  # Cap at 0.95 to keep within bounds
      ax.text(i, y_pos, f"{mean_str}±{std_str}", 
              ha='center', va='bottom', fontsize=7)
  
  # Set font sizes appropriate for paper
  plt.xticks(rotation=30, ha='right', fontsize=8)
  plt.yticks(fontsize=8)
  ax.set_xlabel('Cipher Type', fontsize=9)
  ax.set_ylabel('Score', fontsize=9)
  ax.set_title('Mean Scores by Cipher Type', fontsize=10)
  
  # Add minimal padding to the top of the plot
  ax.margins(y=0.1)  # Reduced margin since we have fixed y-limits
  
  # Adjust layout to minimize whitespace
  plt.tight_layout()
  
  # Save as PDF with minimal whitespace
  plt.savefig('experiment_scores.png')
  plt.savefig('experiment_scores.pdf', format='pdf', bbox_inches='tight', dpi=300)
  print("Plot saved as experiment_scores.pdf")
  
  # Print mapping for reference
  print("\nLabel mapping for reference:")
  for _, row in stats_df.iterrows():
      print(f"{row['short_name']}: {row['experiment_folder']}")

if __name__ == "__main__":
  asyncio.run(main())