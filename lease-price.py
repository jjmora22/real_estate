# Lease Price Prediction Script

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import os
import math

# Set the directory for reports
report_dir = "/Users/jjmora/data_venues/Reports/"
os.makedirs(report_dir, exist_ok=True)

# Load the dataset
print("Step: Loading dataset...")
file_path = '/Users/jjmora/data_venues/DataVenues_data.csv'
data = pd.read_csv(file_path)
print("Step: Dataset loaded successfully.")

# Define normalize_text function
def normalize_text(input_str):
    if isinstance(input_str, (int, float)):
        return str(input_str)
    return ''.join((c for c in unicodedata.normalize('NFD', str(input_str)) if unicodedata.category(c) != 'Mn')).lower()

# Data Cleaning
print("Step: Cleaning data...")
columns_to_drop = ['Fuente', 'Referencia', 'Título', 'Anunciante', 'Empresa', 'Imagen', 'URL', 
                   'Teléfono', 'Email', 'Posible agencia', 'Conservación', 'Comentarios', 
                   'Descartado', 'Estado', 'Ranking', 'Días', 'Demanda', 'Precio unitario',
                   'Latitud', 'Longitud']
data = data.drop(columns=columns_to_drop)
essential_columns = ['Operación', 'Tipología', 'C.P.', 'Municipio', 'Provincia', 
                      'Distrito', 'Barrio', 'Dormitorios', 'Baños', 'Superficie', 'Precio', 'Planta']
data = data.drop_duplicates(subset=essential_columns)

# Process 'Planta' column
print("Step: Processing 'Planta' column...")
data['Planta'] = data['Planta'].fillna('0').replace({'Bajo': '0', 'bajo': '0', 'Ground': '0'})
data['Planta'] = pd.to_numeric(data['Planta'], errors='coerce').fillna(0).astype(int)

# Normalize column names in the dataset
data.columns = [normalize_text(col) for col in data.columns]

# Define and update numerical and categorical variables
numerical_cols = ['dormitorios', 'banos', 'superficie', 'planta']
categorical_cols = ['c.p.', 'operacion', 'tipologia', 'distrito', 'barrio']

# Debugging: Ensure these variables are correctly defined
print("--- Debugging Variable Definitions ---")
print(f"Numerical Columns: {numerical_cols}")
print(f"Categorical Columns: {categorical_cols}")

# Normalize categorical data
print("Step: Normalizing categorical data...")
for col in categorical_cols:
    data[col] = data[col].apply(normalize_text)

print("--- Debugging Column Names ---")
print(data.columns)
print("--- Categorical Columns ---")
print(categorical_cols)
# Debugging Numerical Data during Training
print("\n--- Debugging Numerical Data During Training ---")
print(f"Numerical columns: {numerical_cols}")
print(f"Sample numerical data: {data[numerical_cols].head()}")

# Scale numerical data
scaler = StandardScaler()
numerical_data = scaler.fit_transform(data[numerical_cols])
print(f"Scaled numerical data shape: {numerical_data.shape}")

# Debugging Categorical Data during Training
print("\n--- Debugging Categorical Data During Training ---")
print(f"Categorical columns: {categorical_cols}")
print(f"Sample categorical data: {data[categorical_cols].head()}")

# Encode categorical data
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(data[categorical_cols])
print(f"Encoded categorical feature names: {encoder.get_feature_names_out(categorical_cols)}")
print(f"Encoded categorical data shape: {encoded_features.shape}")

# Combine numerical and categorical data
X = np.hstack([numerical_data, encoded_features])
print(f"Final input shape after training: {X.shape}")


# Define target variable
print("Step: Defining target variable...")
data['Log_Precio'] = np.log1p(data['precio'])

# Add Derived Features
print("Step: Adding new features...")
data['Bedrooms_x_Bathrooms'] = data['dormitorios'] * data['banos']
data['Bathrooms_x_Bedrooms_x_Area'] = data['Bedrooms_x_Bathrooms'] * data['superficie']

# Handle missing values
print("Step: Handling missing values...")
numeric_data = data.select_dtypes(include=['float64', 'int64'])
numeric_cols = numeric_data.columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
data[categorical_cols] = data[categorical_cols].fillna('unknown')

# Define numerical and categorical variables
numerical_cols = ['dormitorios', 'banos', 'superficie', 'planta', 'Bedrooms_x_Bathrooms', 'Bathrooms_x_Bedrooms_x_Area']

# Encode and scale features
print("Step: Encoding and scaling features...")
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(data[categorical_cols])
scaler = StandardScaler()
numerical_data = scaler.fit_transform(data[numerical_cols])
X = np.hstack([numerical_data, encoded_features])

# Model Evaluation
print("Step: Evaluating models...")

def evaluate_models(X, y, original_y=None, label=""):
    print(f"Evaluating with target: {label}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Linear Regression': LinearRegression()
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate MSE in original scale for Log_Precio
        if original_y is not None:
            predictions_original = np.expm1(predictions)
            actuals_original = np.expm1(y_test)
            mse_original = mean_squared_error(actuals_original, predictions_original)
        else:
            mse_original = mean_squared_error(y_test, predictions)

        results[name] = {
            'MAE': mean_absolute_error(y_test, predictions),
            'MSE': mse_original,
            'R2': r2_score(y_test, predictions)
        }
    return models, results


# Run evaluation
log_models, log_comparison = evaluate_models(X, data['Log_Precio'], data['precio'], "Log Transformed Precio")
direct_models, direct_comparison = evaluate_models(X, data['precio'], None, "Direct Precio")


# Selecting the best model
def select_best_model(log_comparison, direct_comparison):
    combined_results = {
        **{f"Log_Precio_{k}": v for k, v in log_comparison.items()},
        **{f"precio_{k}": v for k, v in direct_comparison.items()},
    }
    best_model = min(combined_results.items(), key=lambda x: x[1]['MAE'])
    print(f"Best model: {best_model[0]}")

    if best_model[0].startswith('Log_Precio_'):
        training_target = 'Log_Precio'
        model_name = best_model[0][len('Log_Precio_'):]
    elif best_model[0].startswith('precio_'):
        training_target = 'precio'
        model_name = best_model[0][len('precio_'):]
    else:
        raise ValueError(f"Unknown training target in best model key: {best_model[0]}")

    mse = best_model[1]['MSE']
    print(f"Debugging: Parsed training target: {training_target}, model name: {model_name}, MSE: {mse}")
    return model_name, training_target, mse



# Selecting the best model
model_name, training_target, mse = select_best_model(log_comparison, direct_comparison)


# Debugging outputs for forecasting
print(f"Available models: {list(log_models.keys())}")
print(f"Selected best model: {model_name} trained on {training_target}")


# Process inputs for prediction
# Adjusted `forecast_property` function
# Adjusted MSE transformation
def forecast_property(models, encoder, numerical_cols, categorical_cols, best_model_name):
    print("\nProvide details of the property to forecast the price:")

    # Collect user inputs
    inputs = {}
    for col in numerical_cols:
        if col not in ['Bathrooms_x_Bedrooms_x_Area', 'Bedrooms_x_Bathrooms']:
            inputs[col] = float(input(f"Enter {col.replace('_', ' ')}: "))

    # Automatically calculate derived features
    inputs['Bedrooms_x_Bathrooms'] = inputs['dormitorios'] * inputs['banos']
    inputs['Bathrooms_x_Bedrooms_x_Area'] = inputs['Bedrooms_x_Bathrooms'] * inputs['superficie']

    for col in categorical_cols:
        valid_values = [normalize_text(v) for v in encoder.categories_[categorical_cols.index(col)] if not pd.isnull(v)]
        valid_values.append("unknown")
        while True:
            user_input = input(f"Enter {col} (Valid options: {valid_values}): ").strip()
            normalized_input = normalize_text(user_input)
            if normalized_input in valid_values:
                inputs[col] = normalized_input
                break
            else:
                print(f"Invalid input. Please choose a valid option for {col}.")

    # Transform and debug data
    num_data = scaler.transform([[inputs[col] for col in numerical_cols]])
    print(f"Numerical input (before scaling): {[inputs[col] for col in numerical_cols]}")
    print(f"Numerical data shape during prediction: {num_data.shape}")

    cat_data = encoder.transform([[inputs[col] for col in categorical_cols]])
    print(f"Categorical input (before encoding): {[inputs[col] for col in categorical_cols]}")
    print(f"Categorical data shape during prediction: {cat_data.shape}")
    
    final_input = np.hstack([num_data, cat_data])
    print(f"Final input shape for prediction: {final_input.shape}")

    selected_model = models.get(best_model_name)
    if not selected_model:
        print(f"Error: Model '{best_model_name}' not found.")
        return

    prediction = selected_model.predict(final_input)[0]
    predicted_price = np.expm1(prediction) if 'Log_Precio' in best_model_name else prediction
    print(f"Raw model prediction: {prediction}")
    print(f"Final predicted price: {predicted_price}")

    # Calculate RMSE and suggested price
    rmse = np.sqrt(mse)
    suggested_price = max(0, predicted_price - rmse)


    return predicted_price, rmse, suggested_price


# Function to generate a detailed report
def generate_report(data, log_comparison, direct_comparison, predicted_price, rmse, suggested_price):
    report_file_path = os.path.join(report_dir, "lease_price_report.txt")

    # Generate the correlation matrix with numeric columns only
    correlation_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    heatmap_path = os.path.join(report_dir, "correlation_matrix.png")
    plt.savefig(heatmap_path)
    plt.close()


    # Generate a bar plot for model comparisons
    comparison_df = pd.DataFrame({
        "Model": list(log_comparison.keys()) + list(direct_comparison.keys()),
        "MAE": [v['MAE'] for v in log_comparison.values()] + [v['MAE'] for v in direct_comparison.values()],
        "MSE": [v['MSE'] for v in log_comparison.values()] + [v['MSE'] for v in direct_comparison.values()],
        "R2": [v['R2'] for v in log_comparison.values()] + [v['R2'] for v in direct_comparison.values()],
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(data=comparison_df, x="Model", y="MAE")
    plt.title("Model Comparison: Mean Absolute Error")
    mae_plot_path = os.path.join(report_dir, "mae_comparison.png")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(mae_plot_path)
    plt.close()

    # Generate the text report
    report = f"""
    ### Lease Price Forecast Report ###

    1. Dataset Overview:
       - Total Rows: {len(data)}
       - Columns Used: {', '.join(data.columns)}

    2. Correlation Matrix:
       - Saved as: {heatmap_path}

    3. Model Comparison:
       - Log Transformed Models:
    {pd.DataFrame(log_comparison).T}
       - Direct Models:
    {pd.DataFrame(direct_comparison).T}
       - MAE Bar Plot saved as: {mae_plot_path}

    4. Selected Model:
       - Name: {model_name}
       - RMSE: {rmse:.2f}

    5. Prediction Results:
       - Predicted Price: {predicted_price:.2f}
       - Suggested Competitive Price (1 RMSE Below): {suggested_price:.2f}

    6. Explanation:
       - 1 RMSE Reduction: Balances competitiveness with profitability, appealing to ~67% of comparable offers.
       - 2 RMSE Reduction: Aggressive pricing strategy, appealing to ~95%.

    7. Visual Insights:
       - Correlation Matrix: {heatmap_path}
       - MAE Comparison Plot: {mae_plot_path}
    """

    with open(report_file_path, "w") as file:
        file.write(report)
    print(f"Report saved to {report_file_path}")


# Update the main script to ensure proper variable assignment
# Select best model
model_name, training_target, mse = select_best_model(log_comparison, direct_comparison)

# Forecast property
predicted_price, rmse, suggested_price = forecast_property(
    models=log_models if 'Log_Precio' in model_name else direct_models,
    encoder=encoder,
    numerical_cols=numerical_cols,
    categorical_cols=categorical_cols,
    best_model_name=model_name
)

# Generate the report
generate_report(
    data=data,
    log_comparison=log_comparison,
    direct_comparison=direct_comparison,
    predicted_price=predicted_price,
    rmse=rmse,
    suggested_price=suggested_price
)













