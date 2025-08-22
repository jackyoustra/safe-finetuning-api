# baselines/analyze_judgments.py
import pandas as pd
from config import get_benign_fine_tunes, get_harmful_fine_tunes, RESULTS_DIR
from judgment_utils import (
    analyze_judgments,
    print_results,
    create_visualizations,
    prepare_analysis_data
)

def load_judgments():
    """Load and validate judgments CSV."""
    csv_path = RESULTS_DIR / "judgments.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Judgments file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    required_columns = ['output', 'prompt', 'system_content', 'dataset', 'decision', 'judge_model', 'evaluation_type']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def categorize_models():
    """Get sets of benign and harmful model names."""
    benign_models = {ft.name for ft in get_benign_fine_tunes()}
    harmful_models = {ft.name for ft in get_harmful_fine_tunes()}
    return benign_models, harmful_models

def main():
    # Load judgments
    df = load_judgments()
    
    # Get model categories
    benign_models, harmful_models = categorize_models()
    
    # Analyze results
    results = analyze_judgments(df, benign_models, harmful_models)
    
    # Print results
    print_results(results)
    
    # Prepare data for CSV export
    analysis_data = prepare_analysis_data(results)
    
    # Save to CSV
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df.to_csv(RESULTS_DIR / "judgment_analysis.csv", index=False)
    print(f"\nAnalysis saved to {RESULTS_DIR / 'judgment_analysis.csv'}")
    
    # Create visualizations
    create_visualizations(analysis_df, RESULTS_DIR)

if __name__ == "__main__":
    main() 