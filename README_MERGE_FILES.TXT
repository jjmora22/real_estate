**README: Property Files Merger**

### **Overview**
This script merges multiple property-related CSV files from a specified directory into a single CSV file while preserving column names.

### **How It Works**
1. **Scans the Directory** - It looks for all CSV files in `/Users/jjmora/files_to_merge/`.
2. **Reads the Data** - Loads each CSV file into a DataFrame.
3. **Combines Files** - Merges all data into a single DataFrame.
4. **Saves Output** - Saves the merged data as `merged_properties.csv` in the same directory.

### **How to Use**
1. **Ensure CSV Files Exist**: Place property CSV files in `/Users/jjmora/files_to_merge/`.
2. **Run the Script**:
   ```bash
   python merge_files.py
   ```
3. **Check the Output**: The merged file will be saved as `merged_properties.csv` in the same directory.

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


