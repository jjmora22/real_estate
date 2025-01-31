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
    Filters properties based on user-defined criteria, allowing flexible search options.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    try:
        df = pd.read_csv(input_file)
        
        # Clean 'Planta' column
        df['Planta'] = df['Planta'].apply(clean_floor_value)
        
        # Get user input with defaults
        print("Available Barrios:", df['Barrio'].unique())
        barrio = input("Enter desired Barrio (leave empty for all): ").strip()
        min_floor = input("Enter minimum floor (Planta, leave empty for all): ").strip()
        min_rooms = input("Enter minimum number of rooms (Dormitorios, leave empty for all): ").strip()
        min_baths = input("Enter minimum number of bathrooms (Baños, leave empty for all): ").strip()
        min_surface = input("Enter minimum Superficie (leave empty for all): ").strip()
        min_price = input("Enter minimum price (leave empty for all): ").strip()
        max_price = input("Enter maximum price (leave empty for all): ").strip()
        
        # Apply filters
        if barrio:
            df = df[df['Barrio'].str.lower() == barrio.lower()]
        if min_floor:
            df = df[df['Planta'] >= int(min_floor)]
        if min_rooms:
            df = df[df['Dormitorios'] >= int(min_rooms)]
        if min_baths:
            df = df[df['Baños'] >= int(min_baths)]
        if min_surface:
            df = df[df['Superficie'] >= float(min_surface)]
        if min_price:
            df = df[df['Precio'] >= float(min_price)]
        if max_price:
            df = df[df['Precio'] <= float(max_price)]
        
        # Sort results
        df_sorted = df.sort_values(by=['Precio', 'Superficie', 'Baños', 'Planta', 'Dormitorios'],
                                   ascending=[True, False, False, False, False])
        
        # Save results
        df_sorted.to_csv(output_file, index=False)
        print(f"Filtered properties saved as: {output_file}")
    except Exception as e:
        print(f"Error processing file: {e}")

# Define file paths
input_file = "/Users/jjmora/files_to_merge/cleaned_properties.csv"
output_file = "/Users/jjmora/files_to_merge/custom_filtered_properties.csv"

# Run the function
custom_property_filter(input_file, output_file)
