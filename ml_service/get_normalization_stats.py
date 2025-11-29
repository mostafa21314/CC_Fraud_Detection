#!/usr/bin/env python3
"""
Script to calculate min/max values for average transaction per user and per card
from cleaned_dataset.csv for normalization purposes.
"""

import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("data")
CLEANED_DATASET_PATH = DATA_DIR / "cleaned_dataset.csv"

def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(CLEANED_DATASET_PATH)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few columns: {list(df.columns)[:10]}...")
    
    # Check if columns exist, otherwise calculate from raw data
    all_cols = df.columns.tolist()
    
    # Look for existing columns
    raw_user_col = None
    raw_card_col = None
    
    for col in all_cols:
        col_lower = col.lower()
        if ('avg' in col_lower or 'average' in col_lower) and 'user' in col_lower and 'norm' not in col_lower:
            raw_user_col = col
            break
    
    for col in all_cols:
        col_lower = col.lower()
        if ('avg' in col_lower or 'average' in col_lower) and 'card' in col_lower and 'norm' not in col_lower:
            raw_card_col = col
            break
    
    # If columns don't exist, calculate them from raw data
    if raw_user_col is None:
        print("\n⚠️  'average transaction per user' column not found. Calculating from raw data...")
        print("   Grouping by 'User' and calculating mean of 'Amount_float'...")
        
        # Group by User and calculate mean transaction amount
        user_avg = df.groupby('User')['Amount_float'].mean()
        print(f"   Found {len(user_avg)} unique users")
        raw_user_col = "avg_transaction_per_user_calculated"
        df[raw_user_col] = df['User'].map(user_avg)
    
    if raw_card_col is None:
        print("\n⚠️  'average transaction per card' column not found. Calculating from raw data...")
        print("   Grouping by 'Card' and calculating mean of 'Amount_float'...")
        
        # Group by Card and calculate mean transaction amount
        card_avg = df.groupby('Card')['Amount_float'].mean()
        print(f"   Found {len(card_avg)} unique cards")
        raw_card_col = "avg_transaction_per_card_calculated"
        df[raw_card_col] = df['Card'].map(card_avg)
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    
    # Calculate statistics
    min_user = df[raw_user_col].min()
    max_user = df[raw_user_col].max()
    mean_user = df[raw_user_col].mean()
    std_user = df[raw_user_col].std()
    
    print(f"\nAverage Transaction per User:")
    print(f"  Column: {raw_user_col}")
    print(f"  Min: {min_user:.6f}")
    print(f"  Max: {max_user:.6f}")
    print(f"  Mean: {mean_user:.6f}")
    print(f"  Std: {std_user:.6f}")
    
    min_card = df[raw_card_col].min()
    max_card = df[raw_card_col].max()
    mean_card = df[raw_card_col].mean()
    std_card = df[raw_card_col].std()
    
    print(f"\nAverage Transaction per Card:")
    print(f"  Column: {raw_card_col}")
    print(f"  Min: {min_card:.6f}")
    print(f"  Max: {max_card:.6f}")
    print(f"  Mean: {mean_card:.6f}")
    print(f"  Std: {std_card:.6f}")
    
    # Save the stats
    stats = {
        "avg_transaction_per_user": {
            "min": float(min_user),
            "max": float(max_user),
            "mean": float(mean_user),
            "std": float(std_user),
            "column_name": raw_user_col
        },
        "avg_transaction_per_card": {
            "min": float(min_card),
            "max": float(max_card),
            "mean": float(mean_card),
            "std": float(std_card),
            "column_name": raw_card_col
        }
    }
    
    import json
    output_path = Path("artifacts") / "normalization_stats.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Stats saved to {output_path}")
    print(f"\n{'='*60}")
    print("\nThese min/max values can be used for MinMax normalization:")
    print(f"  normalized_value = (raw_value - min) / (max - min)")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()

