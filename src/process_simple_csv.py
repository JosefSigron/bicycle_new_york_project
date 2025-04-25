import pandas as pd
import sys

def process_csv(input_file, output_file):
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if 'element' column exists
    if 'element' not in df.columns:
        print("Error: 'element' column not found in the CSV")
        return
    
    # Filter rows where 'element' column contains 'TM'
    filtered_df = df[df['element'].astype(str).str.contains('TM', case=True, na=False)].copy()
    
    # Divide values in 'value' column by 10
    if 'value' in filtered_df.columns:
        filtered_df.loc[:, 'value'] = filtered_df['value'] / 10
        # Rename 'value' column to 'temp'
        filtered_df = filtered_df.rename(columns={'value': 'temp'})
    else:
        print("Warning: 'value' column not found in the CSV")
    
    # Drop the 'element' and 'quality_flag' columns
    filtered_df = filtered_df.drop(columns=['element'])
    filtered_df = filtered_df.drop(columns=['quality_flag'])
    
    # Export to CSV
    try:
        filtered_df.to_csv(output_file, index=False)
        print(f"Processed file saved to {output_file}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_csv.py input_file.csv output_file.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_csv(input_file, output_file) 