from pathlib import Path
import json
from typing import List, Dict
import sys
from transformers import AutoTokenizer
import numpy as np
import argparse
from tqdm import tqdm

def analyze_tokens(
    files: List[str],
    tokenizer_name: str,
    fields: List[str] = None,
    batch_size: int = 10000,
) -> Dict:
    """Analyze token lengths for JSONL files."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths = []
    total_tokens = 0
    file_stats = {}

    def batch_tokenize(texts: List[str]) -> List[int]:
        """Tokenize a batch of texts efficiently."""
        if not texts:
            return []
        encodings = tokenizer(texts, add_special_tokens=True, padding=False)
        return [len(ids) for ids in encodings['input_ids']]

    for filepath in tqdm(files, desc="Processing files"):
        file_tokens = 0
        file_sequences = []
        current_batch = []
        sequence_count = 0  # Counter for sequences processed

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())

                    # Determine which fields to process
                    if fields:
                        # Use specified fields
                        text_fields = [item.get(field, '') for field in fields if isinstance(item.get(field, ''), str)]
                    else:
                        # Auto-detect string fields
                        text_fields = [str(value) for value in item.values() if isinstance(value, str)]

                    # Concatenate all text fields into one string per item
                    if text_fields:
                        text = ' '.join(text_fields)
                        current_batch.append(text)
                        sequence_count += 1
                    else:
                        print(f"Warning: No string fields found in item at line {line_num} in {filepath}")

                    # Process batch when it reaches batch_size
                    if len(current_batch) >= batch_size:
                        token_lengths = batch_tokenize(current_batch)
                        lengths.extend(token_lengths)
                        file_sequences.extend(token_lengths)
                        file_tokens += sum(token_lengths)
                        current_batch = []

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line {line_num} in {filepath}")
                    continue

        # Process remaining batch
        if current_batch:
            token_lengths = batch_tokenize(current_batch)
            lengths.extend(token_lengths)
            file_sequences.extend(token_lengths)
            file_tokens += sum(token_lengths)

        # Check if any sequences were processed in this file
        if sequence_count == 0:
            print(f"Warning: No valid sequences found in {filepath}")

        # Store per-file stats
        if file_sequences:
            file_stats[filepath] = {
                "total_tokens": int(file_tokens),
                "num_sequences": len(file_sequences),
                "mean_sequence_length": np.mean(file_sequences),
                "max_sequence_length": max(file_sequences),
                "p95_sequence_length": np.percentile(file_sequences, 95),
            }
        else:
            file_stats[filepath] = {
                "total_tokens": 0,
                "num_sequences": 0,
                "mean_sequence_length": 0,
                "max_sequence_length": 0,
                "p95_sequence_length": 0,
            }

        total_tokens += file_tokens

    # Calculate global stats
    if lengths:
        global_stats = {
            "total_tokens": int(total_tokens),
            "total_sequences": len(lengths),
            "sequence_lengths": lengths,
            "sequence_stats": {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": int(min(lengths)),
                "max": int(max(lengths)),
                "p50": np.percentile(lengths, 50),
                "p95": np.percentile(lengths, 95),
                "p99": np.percentile(lengths, 99),
            },
            "per_file_stats": file_stats,
        }
    else:
        print("Warning: No sequences were processed in total.")
        global_stats = {
            "total_tokens": 0,
            "total_sequences": 0,
            "sequence_lengths": [],
            "sequence_stats": {
                "mean": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0,
            },
            "per_file_stats": file_stats,
        }

    return global_stats

def print_histogram_simple(data: List[int], title: str = "", bins: int = 10, width: int = 50):
    """Print a simple ASCII histogram."""
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = max(hist)
    print(f"\n{title}")
    for i in range(len(hist)):
        bin_range = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
        bar_length = int(hist[i] / max_count * width)
        bar = '█' * bar_length if bar_length > 0 else ''
        print(f"{bin_range:<15} | {bar} {hist[i]}")

def print_stats(stats: Dict, visualize: bool = True) -> None:
    """Pretty print the token statistics."""
    print("\n=== Global Statistics ===")
    print(f"Total Tokens: {stats['total_tokens']:,}")
    print(f"Total Sequences: {stats['total_sequences']:,}")

    seq_stats = stats['sequence_stats']
    print("\n=== Sequence Length Statistics ===")
    print(f"Mean: {seq_stats['mean']:.1f}")
    print(f"Std: {seq_stats['std']:.1f}")
    print(f"Min: {seq_stats['min']}")
    print(f"Max: {seq_stats['max']}")
    print(f"P50: {seq_stats['p50']:.1f}")
    print(f"P95: {seq_stats['p95']:.1f}")
    print(f"P99: {seq_stats['p99']:.1f}")

    if visualize and stats['sequence_lengths']:
        # Print histogram
        print_histogram_simple(stats['sequence_lengths'], title="Sequence Length Histogram", bins=20)
        # Optionally, you can print per-file histograms as well

    print("\n=== Per-File Statistics ===")
    for filepath, file_stats in stats['per_file_stats'].items():
        print(f"\n{Path(filepath).name}:")
        print(f"  Total Tokens: {file_stats['total_tokens']:,}")
        print(f"  Sequences: {file_stats['num_sequences']:,}")
        print(f"  Mean Sequence Length: {file_stats['mean_sequence_length']:.1f}")
        print(f"  Max Sequence Length: {file_stats['max_sequence_length']}")
        print(f"  P95 Sequence Length: {file_stats['p95_sequence_length']:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze token lengths in JSONL datasets')
    parser.add_argument('files', nargs='*', help='JSONL files to analyze')
    parser.add_argument('--tokenizer',
                        default='meta-llama/Meta-Llama-3.1-70B-Instruct',
                        help='Tokenizer to use')
    parser.add_argument('--fields', nargs='+',
                        help='Specific fields to analyze (default: auto-detect string fields)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=10000,
                        help='Batch size for tokenization')
    parser.add_argument('--output',
                        help='Output file for JSON statistics')
    parser.add_argument('--no-visualize',
                        action='store_true',
                        help='Disable histogram visualization')

    args = parser.parse_args()

    if not args.files:
        print("No input files specified")
        sys.exit(1)

    stats = analyze_tokens(
        args.files,
        args.tokenizer,
        fields=args.fields,
        batch_size=args.batch_size,
    )

    # Print stats to console
    print_stats(stats, visualize=not args.no_visualize)

    # Save stats if output file specified
    if args.output:
        # Remove 'sequence_lengths' from stats to reduce file size
        stats_to_save = stats.copy()
        stats_to_save.pop('sequence_lengths', None)
        with open(args.output, 'w') as f:
            json.dump(stats_to_save, f, indent=2)

if __name__ == "__main__":
    main()
