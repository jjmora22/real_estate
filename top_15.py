import pandas as pd
import os

def clean_floor_value(value):
    """Cleans and standardizes the 'Planta' values."""
    if pd.isna(value) or str(value).strip() in ["", "Bajo", "Entresuelo", "Bajos", "Sótano", "Undefined", "Otro"]:
        return 0
    if "A partir del 15" in str(value):
        return 15
    try:
        return int(''.join(filter(str.isdigit, str(value))))
    except ValueError:
        return 0

def get_top_15_opportunities(input_file, output_file):
    """
    Filters the best 15 property options based on user input: number of rooms, bathrooms, and minimum floor.
    Prioritizes the cheapest options first, then the largest by 'Superficie' in case of tie.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    try:
        df = pd.read_csv(input_file)
        
        # Clean 'Planta' column
        df['Planta'] = df['Planta'].apply(clean_floor_value)
        
        # Get user input
        dormitorios = int(input("Enter the number of rooms (Dormitorios): "))
        banos = int(input("Enter the number of bathrooms (Baños): "))
        min_floor = int(input("Enter the minimum floor (Planta, 0 includes all): "))
        
        # Filter properties based on user input
        filtered_df = df[(df['Dormitorios'] == dormitorios) & (df['Baños'] == banos) & (df['Planta'] >= min_floor)]
        
        # Sort by 'Precio' (ascending) and then by 'Superficie' (descending)
        top_15 = filtered_df.sort_values(by=['Precio', 'Superficie'], ascending=[True, False]).head(15)
        
        # Save results to CSV
        top_15.to_csv(output_file, index=False)
        print(f"Top 15 opportunities saved as: {output_file}")
    except Exception as e:
        print(f"Error processing file: {e}")

# Define file paths
input_file = "/Users/jjmora/files_to_merge/cleaned_properties.csv"
output_file = "/Users/jjmora/files_to_merge/top_15_opportunities.csv"

# Run the function
get_top_15_opportunities(input_file, output_file)

