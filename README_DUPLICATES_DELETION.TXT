**README_DUPLICATES_DELETION: Duplicate Removal from Merged Properties**

### **Overview**
This script processes the merged property data sourced from **Data Venues** and removes duplicate entries based on key property attributes.

### **How It Works**
1. **Reads the Merged File** - Loads `merged_properties.csv` from `/Users/jjmora/files_to_merge/`.
2. **Identifies Duplicates** - Detects entries with the same:
   - `Barrio`
   - `Dormitorios`
   - `Baños`
   - `Precio`
3. **Removes Duplicates** - Drops redundant records while keeping the first occurrence.
4. **Saves the Cleaned Data** - Outputs the cleaned version as `cleaned_properties.csv`.

### **How to Use**
1. **Ensure `merged_properties.csv` Exists** - The file should be generated from the merging script.
2. **Run the Script**:
   ```bash
   python remove_duplicates.py
   ```
3. **Check the Output** - The cleaned file will be saved as `cleaned_properties.csv` in the same directory.

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


