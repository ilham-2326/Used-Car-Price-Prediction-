# Used-Car-Price-Prediction-

A machine learning project that predicts used car prices using Linear Regression and a tuned Random Forest Regressor. Built end-to-end: data generation, cleaning, imputation, modelling, hyperparameter tuning, cross-validation, and feature analysis.

## About the Dataset- 
The dataset was generated using an LLM (Claude) and intentionally made messy to simulate real-world data challenges. It contains 1,000 rows across 10 columns:

The dataset includes missing values across multiple columns (~3% per column) and was designed so that price is logically dependent on features — newer cars, lower mileage, premium brands, and fewer owners correspond to higher prices — with added noise for realism.

## Project Highlights
Smart Missing Value Handling
Rather than dropping or mean-imputing the 115 missing engine_size values, a dedicated Random Forest model was trained on the known rows to predict the missing ones. This preserved data integrity instead of introducing bias or losing rows.

**Full ML Pipeline**
- Median imputation for numerical columns
- Most-frequent imputation for categorical columns
- OneHotEncoding for brand, model, fuel type, and transmission
- Proper train/test split with the preprocessor fit only on training data (no data leakage)

### Models
**Linear Regression (Baseline)**
R² Score 0.66
MAE~4,437 AED
RMSE~5,873 AED
A reasonable baseline but limited by its assumption of linear relationships between features and price.

**Random Forest Regressor (Tuned)**
R² Score 0.85
MAE~1,698 AED
RMSE~3,901 AED
CV Mean R² (5-fold)0.87
CV Std Dev0.07

62% reduction in MAE over the baseline. Hyperparameters were tuned using RandomizedSearchCV with 20 iterations and 5-fold cross-validation. 
Best parameters found: n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=2.
Cross-validation confirmed the result was not a lucky split — a mean R² of 0.87 aligns closely with the test R² of 0.85.

## Why Random Forest Outperforms Linear Regression

Captures non-linear relationships (e.g. price depreciation is exponential, not linear)
Handles feature interactions (e.g. mileage matters differently for a 2005 car vs a 2022 car)
Does not assume a straight-line relationship between any feature and price

## Tech Stack
- Python 3.12
- pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, RandomForestRegressor, LinearRegression, RandomizedSearchCV, cross_val_score)
- matplotlib, seaborn

## Limitations

### Synthetic data: 
The dataset was AI-generated. Real-world data would include more noise, outliers, regional pricing quirks, and edge cases. The R² of 0.85 partly reflects the cleaner patterns in synthetic data. The pipeline and methodology are designed to transfer to real datasets with minimal changes.
1,000 rows: A larger real-world dataset would produce more reliable cross-validation estimates and likely surface more interesting feature interactions.
