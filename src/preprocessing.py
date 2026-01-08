import pandas as pd
import json
import os

def load_data(filepath):
    """
    Load data from a jsonl file.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

def preprocess_data(df):
    """
    Clean and preprocess the dataframe.
    """
    # Fill missing text values with empty string
    text_columns = ['description', 'input_description', 'output_description']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Combine text columns
    df['text'] = df['description'] + " " + df['input_description'] + " " + df['output_description']
    
    # Filter out rows with no problem_class or problem_score if any (though problem says they have labels)
    # Check for nulls in target
    if 'problem_class' in df.columns:
        df = df.dropna(subset=['problem_class'])
    if 'problem_score' in df.columns:
        df = df.dropna(subset=['problem_score'])
        
    return df

if __name__ == "__main__":
    input_path = os.path.join("data", "problems_data.jsonl")
    output_path = os.path.join("data", "processed_data.pkl")
    
    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        exit(1)
        
    df = load_data(input_path)
    print(f"Data loaded. Shape: {df.shape}")
    
    print("Preprocessing data...")
    df_clean = preprocess_data(df)
    print(f"Data preprocessed. Shape: {df_clean.shape}")
    
    print("Class distribution:")
    print(df_clean['problem_class'].value_counts())
    
    print(f"Saving processed data to {output_path}...")
    df_clean.to_pickle(output_path)
    print("Done.")
