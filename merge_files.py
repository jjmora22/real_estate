import os
import pandas as pd

def merge_property_files(directory, output_file):
    """
    Merges all CSV files from a given directory into a single file.
    """
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not all_files:
        print("No CSV files found in the directory.")
        return
    
    dataframes = []
    for file in all_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dataframes:
        print("No valid dataframes to merge.")
        return
    
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved as: {output_file}")

# Define the paths
directory = "/Users/jjmora/files_to_merge/"
output_file = "/Users/jjmora/files_to_merge/merged_properties.csv"

# Run the merging function
merge_property_files(directory, output_file)
