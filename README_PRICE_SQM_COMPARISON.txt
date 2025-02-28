# README: Price per Square Meter Comparison Tool

## Overview
This script (`price_sqm_comparison.py`) analyzes real estate property listings to determine which properties have a price per square meter **below the market average**. The goal is to help investors and buyers identify undervalued properties.

## How It Works
1. Loads the dataset: Reads the `cleaned_properties.csv` file containing property listings.
2. Cleans and standardizes data: Handles missing values and normalizes the `Planta` (floor) column.
3. Calculates price per square meter (`Precio_m2`):
   - Computed as `Precio / Superficie` for each property.
4. Computes the average price per square meter** across all properties.
5. Filters out properties** where `Precio_m2` is below the market average.
6. Sorts the filtered properties in ascending order (from the cheapest to the most expensive per square meter).
7. Saves the results** to a CSV file: `properties_below_avg_price_m2.csv`.

## File Structure
- Input File: `cleaned_properties.csv` (Dataset with property listings)
- Output File: `properties_below_avg_price_m2.csv` (Filtered and sorted properties)
- Script: `price_sqm_comparison.py` (Python script that performs the filtering and comparison)

## How to Run the Script
1. Ensure `pandas` is installed:  
   pip install pandas

2. Place `cleaned_properties.csv` in the correct directory.

3. Run the script:
   python price_sqm_comparison.py
  
4. The output file `properties_below_avg_price_m2.csv` will be generated.

## Columns in the Output File
- Precio (€) – Total price of the property.
- Superficie (m²) – Total area of the property.
- Precio_m2 (€ per m²) – Calculated price per square meter.
- (Includes other original columns from `cleaned_properties.csv`)

## Use Cases
- Investment Analysis: Find properties with better price opportunities.
- Market Insights: Understand pricing trends per square meter.
- Decision Making: Identify properties with potential for appreciation.
