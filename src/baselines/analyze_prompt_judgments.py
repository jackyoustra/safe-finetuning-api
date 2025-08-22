# baselines/analyze_prompt_judgments.py
import pandas as pd
from config import get_benign_fine_tunes, get_harmful_fine_tunes, RESULTS_DIR
from judgment_utils import (
    analyze_judgments,
    print_results,
    create_visualizations,
    prepare_analysis_data
)

def load_judgments():
    """Load and validate prompt judgments CSV."""
    csv_path = RESULTS_DIR / "prompt_judgments.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Prompt judgments file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    required_columns = ['prompt', 'system_content', 'dataset', 'decision', 'judge_model', 'evaluation_type', 'harm_category']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def categorize_models():
    """Get sets of benign and harmful model names."""
    benign_models = {ft.name for ft in get_benign_fine_tunes()}
    harmful_models = {ft.name for ft in get_harmful_fine_tunes()}
    return benign_models, harmful_models

def create_prompt_summary(df):
    """Create a simplified summary of unique prompts and their properties."""
    # Get only the columns we care about
    prompt_df = df[['prompt', 'dataset', 'harm_category', 'decision', 'evaluation_type']].copy()
    
    # Keep only SAFE/UNSAFE/PARSE ERROR decisions
    prompt_df = prompt_df[prompt_df['decision'].isin(['SAFE', 'UNSAFE', 'PARSE ERROR'])]
    
    # Drop duplicates to get unique prompt entries
    prompt_df = prompt_df.drop_duplicates()
    
    return prompt_df

def main():
    # Load judgments
    df = load_judgments()
    
    # Get model categories
    # benign_models, harmful_models = categorize_models()
    
    # Analyze results
    results = analyze_judgments(df) #, benign_models, harmful_models)
    
    # Print results with prefix to distinguish from output analysis
    print_results(results, prefix="Prompt ")
    
    # Create and save prompt summary
    prompt_summary = create_prompt_summary(df)
    prompt_summary.to_csv(RESULTS_DIR / "prompt_summary.csv", index=False)
    print(f"\nPrompt summary saved to {RESULTS_DIR / 'prompt_summary.csv'}")
    
    # Prepare data for CSV export
    analysis_data = prepare_analysis_data(results)
    
    # Save to CSV
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df.to_csv(RESULTS_DIR / "prompt_judgment_analysis.csv", index=False)
    print(f"\nAnalysis saved to {RESULTS_DIR / 'prompt_judgment_analysis.csv'}")
    
    # Create visualizations with prefix to distinguish from output visualizations
    create_visualizations(analysis_df, RESULTS_DIR, prefix="prompt_", title_prefix="Prompt ")

if __name__ == "__main__":
    main() 