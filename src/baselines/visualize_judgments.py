import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Set style configurations
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (12, 6),
})

def load_data():
    """Load the judgment analysis data."""
    results_dir = Path("results")
    df = pd.read_csv(results_dir / "judgment_analysis.csv")
    return df

def create_accuracy_distribution(df, output_dir):
    """Create violin plots showing accuracy distribution by model type and evaluation type."""
    # Filter out OVERALL rows and get only rows with accuracy
    plot_df = df[
        (df['dataset'] != 'OVERALL') & 
        pd.notna(df['accuracy'])
    ].copy()
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=plot_df, x='type', y='accuracy', hue='evaluation_type', split=True)
    plt.title('Accuracy Distribution by Model Type and Evaluation Type')
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_decision_heatmap(df, output_dir):
    """Create heatmap showing decision patterns."""
    # Get decision columns
    decision_cols = [col for col in df.columns if col.startswith('decision_')]
    if not decision_cols:
        return
    
    # Prepare data for each evaluation type
    for eval_type in df['evaluation_type'].unique():
        judge_data = df[
            (df['dataset'] != 'OVERALL') & 
            (df['evaluation_type'] == eval_type)
        ]
        
        # Create pivot table
        pivot_data = pd.pivot_table(
            judge_data,
            values=decision_cols,
            index='type',
            aggfunc='mean'
        )
        
        # Normalize the data
        row_sums = pivot_data.sum(axis=1)
        pivot_data = pivot_data.div(row_sums, axis=0)
        
        plt.figure(figsize=(10, 4))
        sns.heatmap(pivot_data, annot=True, fmt='.2%', cmap='RdYlGn', center=0.5)
        plt.title(f'Decision Pattern Heatmap - {eval_type}')
        plt.tight_layout()
        plt.savefig(output_dir / f'decision_heatmap_{eval_type}.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_judge_performance(df, output_dir):
    """Create bar plot showing judge model performance."""
    # Get overall performance for each judge
    judge_perf = df[df['dataset'] == 'OVERALL'].copy()
    
    # Calculate combined accuracy
    judge_perf['combined_accuracy'] = judge_perf.apply(
        lambda x: (
            (x['benign_accuracy'] * x['benign_samples'] if pd.notna(x['benign_accuracy']) else 0) +
            (x['harmful_accuracy'] * x['harmful_samples'] if pd.notna(x['harmful_accuracy']) else 0)
        ) / (
            (x['benign_samples'] if pd.notna(x['benign_samples']) else 0) +
            (x['harmful_samples'] if pd.notna(x['harmful_samples']) else 0)
        ),
        axis=1
    )
    
    # Sort by combined accuracy
    judge_perf = judge_perf.sort_values('combined_accuracy', ascending=False)
    
    plt.figure(figsize=(15, 6))
    sns.barplot(data=judge_perf, x='judge_model', y='combined_accuracy', 
                hue='evaluation_type', dodge=False)
    plt.xticks(rotation=45, ha='right')
    plt.title('Overall Judge Model Performance')
    plt.xlabel('Judge Model')
    plt.ylabel('Combined Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'judge_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dataset_difficulty(df, output_dir):
    """Create scatter plot showing dataset difficulty."""
    # Filter for non-OVERALL rows with accuracy
    dataset_diff = df[
        (df['dataset'] != 'OVERALL') & 
        pd.notna(df['accuracy'])
    ].groupby(['dataset', 'type', 'evaluation_type'])['accuracy'].agg(['mean', 'std']).reset_index()
    
    # Sort by mean accuracy
    dataset_diff = dataset_diff.sort_values('mean', ascending=True)
    
    plt.figure(figsize=(20, 10))  # Increased figure size
    
    # Create scatter plot with different markers for evaluation types
    for eval_type in dataset_diff['evaluation_type'].unique():
        mask = dataset_diff['evaluation_type'] == eval_type
        plt.scatter(
            range(sum(mask)), 
            dataset_diff[mask]['mean'],
            c=dataset_diff[mask]['type'].map({'benign': 'green', 'harmful': 'red'}),
            marker='o' if eval_type == 'self-reflection' else ('^' if eval_type == 'frontier' else 's'),
            alpha=0.6,
            label=eval_type,
            s=100  # Increased marker size
        )
        plt.errorbar(
            range(sum(mask)),
            dataset_diff[mask]['mean'],
            yerr=dataset_diff[mask]['std'],
            fmt='none',
            alpha=0.2
        )
    
    plt.xticks(range(len(dataset_diff)), dataset_diff['dataset'], rotation=45, ha='right')
    plt.legend(title='Evaluation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Dataset Difficulty (Mean Accuracy with Standard Deviation)')
    plt.xlabel('Dataset')
    plt.ylabel('Mean Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_difficulty.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_post_attack_heatmap(df, output_dir):
    """Create heatmaps showing post-attack accuracies."""
    # Filter and prepare data
    attack_data = df[
        (df['dataset'] != 'OVERALL') & 
        pd.notna(df['accuracy'])
    ].copy()
    
    def style_average_columns(pivot_data, ax, title, figsize=(20, 6)):
        plt.figure(figsize=figsize)
        
        # Set up the plot with gridlines
        ax = plt.gca()
        ax.set_axisbelow(True)
        ax.grid(True, color='gray', linestyle='-', linewidth=0.2, alpha=0.5)
        
        # Create heatmap with diverging colormap
        hm = sns.heatmap(pivot_data * 100,
                    annot=True, 
                    fmt='.0f',
                    cmap='RdYlGn',
                    vmin=pivot_data.min().min() * 100,
                    vmax=pivot_data.max().max() * 100,
                    center=(pivot_data.min().min() * 100 + pivot_data.max().max() * 100) / 2,
                    cbar_kws={'label': 'Accuracy'},
                    linewidths=0.5,
                    linecolor='white',
                    annot_kws={'size': 8},
                    cbar=True)
        
        # Style the average and geomean columns
        last_cols = 2  # Number of special columns (AVERAGE and GEOMEAN)
        n_cols = len(pivot_data.columns)
        
        # Add divider line before the special columns
        plt.axvline(x=n_cols-last_cols, color='black', linewidth=2)
        
        # Add hatching patterns to the special columns
        for i in range(len(pivot_data.index)):
            # Average column (diagonal hatching)
            hm.add_patch(plt.Rectangle((n_cols-2, i), 1, 1, fill=True, 
                                     facecolor='white', alpha=0.2,
                                     hatch='///', edgecolor='gray'))
            # Geomean column (crossed hatching)
            hm.add_patch(plt.Rectangle((n_cols-1, i), 1, 1, fill=True,
                                     facecolor='white', alpha=0.2,
                                     hatch='xx', edgecolor='gray'))
        
        # Make special columns text bold and larger
        for text in hm.texts:
            col_idx = text.get_position()[0]
            if col_idx >= n_cols - last_cols - 0.5:  # If in special columns
                text.set_weight('bold')
                text.set_size(10)
        
        # Style the column labels
        for i in range(last_cols):
            ax.get_xticklabels()[-(i+1)].set_weight('bold')
            ax.get_xticklabels()[-(i+1)].set_size(12)
        
        plt.title(title, pad=20)
        plt.xlabel('Dataset', labelpad=10)
        plt.ylabel('Evaluation Type', labelpad=10)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        return hm
    
    # Version 1: All fine-tunes
    pivot_data = pd.pivot_table(
        attack_data,
        values='accuracy',
        index='evaluation_type',
        columns='dataset',
        aggfunc='mean'
    )
    
    # Add average and geometric mean columns
    other_cols = sorted([col for col in pivot_data.columns if col not in ['AVERAGE', 'GEOMEAN']])
    pivot_data['AVERAGE'] = pivot_data[other_cols].mean(axis=1)
    pivot_data['GEOMEAN'] = np.exp(np.log(pivot_data[other_cols]).mean(axis=1))
    pivot_data = pivot_data[other_cols + ['AVERAGE', 'GEOMEAN']]
    
    # Create first heatmap
    hm1 = style_average_columns(pivot_data, plt.gca(), 'Post-Attack Accuracies by Evaluation Type (All Datasets)')
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_dir / 'post_attack_heatmap_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Version 2: Averaged MMLU results
    attack_data['is_mmlu'] = attack_data['dataset'].str.startswith('mmlu_')
    
    # Create new dataframe with MMLU averaged and other datasets as is
    mmlu_avg = attack_data[attack_data['is_mmlu']].groupby(['evaluation_type'])['accuracy'].mean()
    non_mmlu = attack_data[~attack_data['is_mmlu']].groupby(['evaluation_type', 'dataset'])['accuracy'].mean().unstack()
    
    # Combine MMLU average with other datasets
    combined_data = non_mmlu.copy()
    combined_data['mmlu_average'] = mmlu_avg
    
    # Add arithmetic and geometric means
    other_cols = [col for col in combined_data.columns if col not in ['AVERAGE', 'GEOMEAN']]
    combined_data['AVERAGE'] = combined_data[other_cols].mean(axis=1)
    combined_data['GEOMEAN'] = np.exp(np.log(combined_data[other_cols]).mean(axis=1))
    
    # Sort columns to put MMLU average first and summary columns last
    cols = ['mmlu_average'] + [col for col in other_cols if col != 'mmlu_average'] + ['AVERAGE', 'GEOMEAN']
    combined_data = combined_data[cols]
    
    # Create second heatmap
    hm2 = style_average_columns(combined_data, plt.gca(), 
                             'Post-Attack Accuracies by Evaluation Type (MMLU Averaged)',
                             figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_dir / 'post_attack_heatmap_mmlu_avg.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory
    output_dir = Path("results/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Create visualizations
    create_accuracy_distribution(df, output_dir)
    create_decision_heatmap(df, output_dir)
    create_judge_performance(df, output_dir)
    create_dataset_difficulty(df, output_dir)
    create_post_attack_heatmap(df, output_dir)
    
    print("Visualizations have been created in:", output_dir)

if __name__ == "__main__":
    main() 