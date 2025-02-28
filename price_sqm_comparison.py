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

def custom_property_filter(input_file, output_file):
    """
    Filters properties based on price per square meter (Precio_m2),
    selecting only those with a value below the overall average.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    try:
        df = pd.read_csv(input_file)
        
        # Clean 'Planta' column
        df['Planta'] = df['Planta'].apply(clean_floor_value)

        # Ensure necessary columns exist
        if 'Precio' not in df.columns or 'Superficie' not in df.columns:
            print("Error: Missing required columns 'Precio' or 'Superficie'.")
            return

        # Remove rows where 'Superficie' is zero or null to avoid division errors
        df = df[df['Superficie'] > 0]

        # Calculate price per square meter
        df['Precio_m2'] = df['Precio'] / df['Superficie']

        # Compute the average price per square meter
        precio_m2_promedio = df['Precio_m2'].mean()
        print(f"Precio promedio por m²: {precio_m2_promedio:.2f}")

        # Filter properties with a price per square meter below the average
        df_filtered = df[df['Precio_m2'] < precio_m2_promedio]

        # Sort the results from the lowest to the highest price per square meter
        df_sorted = df_filtered.sort_values(by='Precio_m2', ascending=True)

        # Save the results
        df_sorted.to_csv(output_file, index=False)
        print(f"Filtered properties saved as: {output_file}")

    except Exception as e:
        print(f"Error processing file: {e}")

# Define file paths
input_file = "/Users/jjmora/files_to_merge/cleaned_properties.csv"
output_file = "/Users/jjmora/files_to_merge/properties_below_avg_price_m2.csv"

# Run the function
custom_property_filter(input_file, output_file)

