import pandas as pd
import os

def remove_duplicates(input_file, output_file):
    """
    Reads the merged CSV file and removes duplicates based on 'Barrio', 'Dormitorios', 'Baños', and 'Precio'.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    try:
        df = pd.read_csv(input_file)
        df_cleaned = df.drop_duplicates(subset=['Barrio', 'Dormitorios', 'Baños', 'Precio'])
        df_cleaned.to_csv(output_file, index=False)
        print(f"Duplicates removed. Cleaned file saved as: {output_file}")
    except Exception as e:
        print(f"Error processing file: {e}")

# Define file paths
input_file = "/Users/jjmora/files_to_merge/merged_properties.csv"
output_file = "/Users/jjmora/files_to_merge/cleaned_properties.csv"

# Run the function
remove_duplicates(input_file, output_file)

