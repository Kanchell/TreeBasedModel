# XGBoost Regression Analysis  

## Overview  
This project implements an **XGBoost regression pipeline** with preprocessing, model training, evaluation, and visualization. It includes functions for:  

- Feature and target scaling  
- Splitting the dataset into training, validation, and test sets  
- Training an XGBoost Regressor  
- Evaluating performance using **MSE, MAE, RMSE, and R²**  
- Plotting predicted vs actual values and residuals  
- Testing multiple hyperparameter configurations to optimize performance  

The script is designed for experimentation with **tabular datasets** containing features x1, x2, x3, x4 and target variables y1.  

---

## File Description  
- **decisionTree.py**: Main script containing data preparation, model training, evaluation, and visualization functions.  

---

## Requirements  
Make sure you have the following Python libraries installed:  

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn
```

---

## Usage  

1. Place your dataset (`dataset.csv`) in the **data** folder, or adjust the path in the script.  
2. Run the script:  

```bash
python decisionTree.py
```

3. The program will:  
   - Load and preprocess the dataset  
   - Train an XGBoost regression model  
   - Print metrics for training, validation, and test sets  
   - Display visualizations (scatter plots and residuals)  
   - Optionally test different hyperparameter configurations  

---

## Functions  

- **`x_scale(x, p=7.5)`**: Scales features with a log-based transformation.  
- **`y_scale(y)`**: Scales target variable (handles negatives).  
- **`prepare_data(df, target="y1")`**: Cleans and transforms the dataset.  
- **`dataSplit(df)`**: Splits data into train/validation/test sets.  
- **`model(df, target="y1")`**: Trains and evaluates the XGBoost model.  
- **`create_plots(y_test, y_test_pred, y_val, y_val_pred, target)`**: Generates evaluation plots.  
- **`test_different_parameters(df, target="y1")`**: Tests multiple hyperparameter combinations.  

---

## Example Output  

- Training, validation, and test metrics printed in the console.  
- Visualization plots comparing predicted vs actual values and residuals.  
- Best hyperparameter settings identified for improved model performance.  

---

## Author  
**Nisha Madavaprasad**  
