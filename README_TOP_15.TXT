**README_TOP_15: Top 15 Property Opportunities**

### **Overview**
This script processes the cleaned property data from **cleaned_properties.csv** and filters the top 15 best opportunities based on user preferences.

### **How It Works**
1. **Loads the Cleaned Data** - Reads `cleaned_properties.csv` from `/Users/jjmora/files_to_merge/`.
2. **User Input Requirements** - Asks for:
   - Number of rooms (`Dormitorios`).
   - Number of bathrooms (`Baños`).
   - Minimum floor (`Planta`), where:
     - `0` includes all floors.
     - `2` includes only properties from floor `2` and above.
3. **Cleans Floor Data (`Planta`)** - Converts values like:
   - **'Bajo', 'Entresuelo', 'Bajos', empty values → 0**.
   - **Ordinal values like '3º' → 3**.
   - **'A partir del 15º' → 15**.
   - Removes non-numeric entries like `'Otro'`, `'Undefined'`, and `'Sótano'`.
4. **Filters Matching Properties** - Keeps only those that match the user’s criteria.
5. **Sorts & Selects Top 15** - Prioritizes by:
   - **Lowest 'Precio' (cheapest first).**
   - **Largest 'Superficie' (if prices match).**
6. **Saves the Results** - Outputs the top 15 matches as `top_15_opportunities.csv`.

### **How to Use**
1. **Ensure `cleaned_properties.csv` Exists** - The file should be generated from the duplicate removal script.
2. **Run the Script**:
   ```bash
   python filter_top_15.py
   ```
3. **Check the Output** - The filtered list will be saved as `top_15_opportunities.csv` in the same directory.

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
