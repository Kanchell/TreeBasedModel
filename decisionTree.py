import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def x_scale(x, p=7.5):
    """Scale feature x using log transformation."""
    return 1/p * np.log(1 + x * (np.exp(p) - 1))

def y_scale(y):
    """Scale target variable using log transform (handles negatives)."""
    return np.log(1 + y) if y >= 0 else -np.log(1 - y)


def prepare_data(df, target="y1"):
    # Check for missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print("No missing values in the dataset.")
    else:
        print("Missing values detected:\n", missing[missing > 0])

    # Apply transformations
    df = df.copy()
    df['x1'] = df['x1'].apply(lambda x: x_scale(x, p=7.5))
    df[target] = df[target].apply(y_scale)

    return df

def dataSplit(df):
    """Split dataframe into train/validation/test sets using np.split"""
    train, validate, test = np.split(df.sample(frac=1, random_state=42), 
                                   [int(.6*len(df)), int(.8*len(df))])
    return train, validate, test

def model(df, target="y1"):
    feature_cols = ['x1', 'x2', 'x3', 'x4']

    # Split the data
    train_df, val_df, test_df = dataSplit(df)

    # Prepare features and targets
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df[target]
    y_val = val_df[target]
    y_test = test_df[target]

    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # XGBoost Regressor - SIMPLE VERSION FOR OLD XGBOOST
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42
    )

    # SIMPLE FIT - NO EXTRA PARAMETERS
    print("Training XGBoost model")
    model.fit(X_train, y_train)
    print("Training completed!")

    # Predictions on all sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calculate metrics for all sets
    def print_metrics(y_true, y_pred, set_name):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"\n{set_name} Metrics:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²:   {r2:.6f}")
        return mse, mae, rmse, r2

    # Print metrics for all sets
    train_metrics = print_metrics(y_train, y_train_pred, "Training")
    val_metrics = print_metrics(y_val, y_val_pred, "Validation")
    test_metrics = print_metrics(y_test, y_test_pred, "Test")

    
    # Plotting
    create_plots(y_test, y_test_pred, y_val, y_val_pred, target)

    return model

def create_plots(y_test, y_test_pred, y_val, y_val_pred, target):
    """Create visualization plots"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'XGBoost Regression Results - {target}', fontsize=16, fontweight='bold')

    # Plot 1: Test Set - Predicted vs Actual
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, color='blue', s=20)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Test Set: Predicted vs Actual')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Set - Predicted vs Actual
    axes[0, 1].scatter(y_val, y_val_pred, alpha=0.6, color='green', s=20)
    min_val = min(y_val.min(), y_val_pred.min())
    max_val = max(y_val.max(), y_val_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title('Validation Set: Predicted vs Actual')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Test Residuals
    test_residuals = y_test - y_test_pred
    axes[1, 0].scatter(y_test_pred, test_residuals, alpha=0.6, color='orange', s=20)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Test Set: Residuals Plot')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Validation Residuals
    val_residuals = y_val - y_val_pred
    axes[1, 1].scatter(y_val_pred, val_residuals, alpha=0.6, color='purple', s=20)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Validation Set: Residuals Plot')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def test_different_parameters(df, target="y1"):
    """Test different hyperparameters to find better settings"""
    
    feature_cols = ['x1', 'x2', 'x3', 'x4']
    train_df, val_df, test_df = dataSplit(df)
    
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    y_train = train_df[target]
    y_val = val_df[target]
    
    # Different parameter combinations to test
    param_combinations = [
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3},
        {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 4},
        {'n_estimators': 200, 'learning_rate': 0.08, 'max_depth': 4},
        {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 3},
    ]
    
    best_score = float('inf')
    best_params = None
    results = []
    
    print("Testing different parameter combinations...")
    
    for i, params in enumerate(param_combinations, 1):
        print(f"Testing combination {i}: {params}")
        
        model = XGBRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        results.append({
            'params': params,
            'val_mse': val_mse,
            'val_r2': val_r2
        })
        
        print(f"  Validation MSE: {val_mse:.6f}, R²: {val_r2:.6f}")
        
        if val_mse < best_score:
            best_score = val_mse
            best_params = params
        
        print()
    
    print(f"BEST PARAMETERS: {best_params}")
    print(f"BEST VALIDATION MSE: {best_score:.6f}")
    
    return best_params, results

if __name__ == "__main__":
    print("XGBoost Regression Analysis")
    
    # Load dataset
    df = pd.read_csv(" http://129.10.224.71/~apaul/data/tests/dataset.csv.gz")
    print(f"Original dataset size: {len(df)} rows")
    
    # Sample data for large dataw
    if len(df) > 500000:
        df = df.sample(n=500000, random_state=42)
        print(f"Sampled dataset to {len(df)} rows")

    # Choose target column
    target = "y1"  # Change to "y2" if needed
    print(f"Target variable: {target}")

    # Preprocess data
    print("\nPreprocessing data...")
    df = prepare_data(df, target)
    
    # Train basic model
    print("TRAINING XGBOOST MODEL")
    
    trained_model = model(df, target)