**README_LEASE_PRICE: Lease Price Prediction**

### **Overview**
This script predicts lease prices for properties using machine learning models trained on data sourced from **Data Venues**. The model evaluates various property features and generates price forecasts, along with a competitive pricing recommendation.

### **How It Works**
1. **Loads and Cleans Data** - Reads `DataVenues_data.csv` from `/Users/jjmora/data_venues/`.
2. **Data Preprocessing**:
   - Drops unnecessary columns (`Fuente`, `Referencia`, etc.).
   - Standardizes categorical variables (`Barrio`, `Distrito`, etc.).
   - Normalizes `Planta` (floor level) values:
     - **'Bajo', 'Entresuelo', '0' → 0**
     - **Numeric and ordinal values cleaned appropriately**
   - Handles missing values and feature engineering.
3. **Feature Encoding & Scaling**:
   - Categorical variables one-hot encoded.
   - Numerical variables standardized.
4. **Trains and Evaluates Models**:
   - Uses **Random Forest**, **Gradient Boosting**, and **Linear Regression**.
   - Compares models using **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**.
   - Selects the best-performing model.
5. **Price Prediction & Competitive Strategy**:
   - Predicts lease price for user-defined property details.
   - Suggests a **competitive pricing strategy** (adjusting for RMSE).
6. **Generates Reports & Insights**:
   - Creates a **correlation matrix**.
   - Compares models in a **bar chart**.
   - Saves a **detailed price prediction report**.

### **How to Use**
1. **Ensure `DataVenues_data.csv` Exists** - The dataset must be present in `/Users/jjmora/data_venues/`.
2. **Run the Script**:
   ```bash
   python lease-price.py
   ```
3. **Check the Output**:
   - The predicted lease price will be printed.
   - Reports and visualizations will be saved in `/Users/jjmora/data_venues/Reports/`.

### **Requirements**
- Python 3.x
- Libraries: Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib, Seaborn
  ```bash
  pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
  ```

### **Git Repository**
This script is stored in the GitHub repository:
[https://github.com/jjmora22/real_estate](https://github.com/jjmora22/real_estate)

To clone the repository:
```bash
git clone git@github.com:jjmora22/real_estate.git
```

### **Author**
**Juan José de la Mora**


