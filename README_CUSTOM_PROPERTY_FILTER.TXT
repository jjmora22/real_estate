**README_CUSTOM_PROPERTY_FILTER: Flexible Property Filtering**

### **Overview**
This script allows users to filter properties from **Data Venues** based on flexible search criteria, selecting only relevant opportunities from `cleaned_properties.csv`.

### **How It Works**
1. **Loads the Cleaned Data** - Reads `cleaned_properties.csv` from `/Users/jjmora/files_to_merge/`.
2. **User Input Parameters** - Asks for:
   - `Barrio` (Shows all available options, accepts specific input, leave empty for all).
   - `Planta` (Minimum floor, leave empty for all floors).
   - `Dormitorios` (Minimum number of rooms, leave empty for all).
   - `Baños` (Minimum number of bathrooms, leave empty for all).
   - `Superficie` (Minimum area in square meters, leave empty for all).
   - `Precio Min` (Minimum price, leave empty for all).
   - `Precio Max` (Maximum price, leave empty for all).
3. **Cleans and Standardizes 'Planta' Values** - Handles cases such as:
   - **'Bajo', 'Entresuelo', 'Bajos', empty values → 0**.
   - **Ordinal values like '3º' → 3**.
   - **'A partir del 15º' → 15**.
   - Filters out non-numeric values (`'Otro'`, `'Undefined'`, `'Sótano'`).
4. **Applies Filters** - Keeps properties matching user-defined criteria.
5. **Sorts & Selects Best Matches** - Prioritizes results by:
   - **Lowest 'Precio' (cheapest first).**
   - **Largest 'Superficie' (if prices match).**
   - **Most 'Baños' (if prices & area match).**
   - **Highest 'Planta' (if previous conditions match).**
   - **Most 'Dormitorios' (as a final tiebreaker).**
6. **Saves the Results** - Outputs the filtered properties as `custom_filtered_properties.csv`.

### **How to Use**
1. **Ensure `cleaned_properties.csv` Exists** - The file should be generated from the duplicate removal script.
2. **Run the Script**:
   ```bash
   python custom_property_filter.py
   ```
3. **Check the Output** - The filtered results will be saved as `custom_filtered_properties.csv` in the same directory.

### **Requirements**
- Python 3.x
- Pandas library (install with `pip install pandas` if needed)

### **Git Repository**
This script is stored in the GitHub repository:
[https://github.com/jjmora22/real_estate](https://github.com/jjmora22/real_estate)

To clone the repository:
```bash
git clone git@github.com:jjmora22/real_estate.git
```

### **Author**
**Juan José de la Mora**
