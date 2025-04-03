from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import os
import matplotlib.pyplot as plt
from common_utils import retrieveDataset, normalize_data
from tqdm import tqdm
import logging
import pandas as pd
import seaborn as sns
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_types = ['SMRI', 'DTI']
subset_types = ['FTPC', 'FTP', 'FTC']

# Initialize a dictionary to store all results
all_results = {
    'Data Type': [],
    'Subset Type': [],
    'Target': [],
    'Mean Test R2': [],
    'Std Test R2': [],
    'Mean Test Correlation': [],
    'Std Test Correlation': [],
    'Mean Train R2': [],
    'Std Train R2': [],
    'Mean Train Correlation': [],
    'Std Train Correlation': [],
    'Mean Validation R2': [],
    'Std Validation R2': [],
    'Mean Validation Correlation': [],
    'Std Validation Correlation': []
}

# Initialize a dictionary to store feature importance
feature_importance = {
    'Data Type': [],
    'Subset Type': [],
    'Target': [],
    'Fold': [],
    'Feature': [],
    'Coefficient': []
}

# Determine the number of available CPU cores
num_cores = multiprocessing.cpu_count()

for data_type in data_types:
    logging.info(f"Starting training for {data_type}")
    for subset_type in subset_types:
        logging.info(f"Training for subset {subset_type}")
        
        # Retrieve and normalize dataset
        dataset = retrieveDataset(data=data_type, subset=subset_type)
        
        for target, (X_pd, y_pd) in dataset.items():
            logging.info(f"Target: {target}")
            
            # Normalize the data
            X, y = normalize_data(X_pd, y_pd)
            
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32).ravel()

            # Define the outer 5-fold cross-validation
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

            # Initialize the Bayesian Ridge Regression model with default parameters
            model = BayesianRidge()

            # Store metrics from outer folds
            test_r2_scores = []
            test_correlations = []
            train_r2_scores = []
            train_correlations = []
            validation_r2_scores = []
            validation_correlations = []

            logging.info("Starting cross-validation process...\n")

            # Outer loop of 5-fold cross-validation with tqdm progress bar
            for outer_fold, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(X), total=5, desc="Outer CV Folds"), 1):
                logging.info(f"\nProcessing outer fold {outer_fold}...")

                # Split the data into training and testing sets
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Fit the model on the training data
                model.fit(X_train, y_train)

                # Predict on training and testing data
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Calculate R² scores
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)

                # Calculate Pearson correlation coefficients
                train_correlation, _ = pearsonr(y_train, y_pred_train)
                test_correlation, _ = pearsonr(y_test, y_pred_test)

                # Store the metrics
                train_r2_scores.append(train_r2)
                test_r2_scores.append(test_r2)
                train_correlations.append(train_correlation)
                test_correlations.append(test_correlation)

                # Calculate validation predictions using cross_val_predict
                # Here, validation is performed on the training data using inner cross-validation
                y_val_pred = cross_val_predict(model, X_train, y_train, cv=outer_cv, n_jobs=num_cores-1)

                # Calculate R² and Pearson correlation for validation
                val_r2 = r2_score(y_train, y_val_pred)
                val_correlation, _ = pearsonr(y_train, y_val_pred)

                validation_r2_scores.append(val_r2)
                validation_correlations.append(val_correlation)

                logging.info(
                    f"Completed outer fold {outer_fold}. "
                    f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, "
                    f"Train Correlation: {train_correlation:.4f}, Test Correlation: {test_correlation:.4f}, "
                    f"Validation R²: {val_r2:.4f}, Validation Correlation: {val_correlation:.4f}"
                )

                # **Extract and Store Coefficients with Actual Feature Names**
                coefficients = model.coef_
                feature_names = X_pd.columns  # Actual feature names from DataFrame

                for feature, coef in zip(feature_names, coefficients):
                    feature_importance['Data Type'].append(data_type)
                    feature_importance['Subset Type'].append(subset_type)
                    feature_importance['Target'].append(target)
                    feature_importance['Fold'].append(outer_fold)
                    feature_importance['Feature'].append(feature)
                    feature_importance['Coefficient'].append(coef)

            # Calculate mean and std for all metrics
            mean_test_r2 = np.mean(test_r2_scores)
            std_test_r2 = np.std(test_r2_scores)
            mean_test_correlation = np.mean(test_correlations)
            std_test_correlation = np.std(test_correlations)

            mean_train_r2 = np.mean(train_r2_scores)
            std_train_r2 = np.std(train_r2_scores)
            mean_train_correlation = np.mean(train_correlations)
            std_train_correlation = np.std(train_correlations)

            mean_val_r2 = np.mean(validation_r2_scores)
            std_val_r2 = np.std(validation_r2_scores)
            mean_val_correlation = np.mean(validation_correlations)
            std_val_correlation = np.std(validation_correlations)

            # Store the aggregated results
            all_results['Data Type'].append(data_type)
            all_results['Subset Type'].append(subset_type)
            all_results['Target'].append(target)
            all_results['Mean Test R2'].append(mean_test_r2)
            all_results['Std Test R2'].append(std_test_r2)
            all_results['Mean Test Correlation'].append(mean_test_correlation)
            all_results['Std Test Correlation'].append(std_test_correlation)
            all_results['Mean Train R2'].append(mean_train_r2)
            all_results['Std Train R2'].append(std_train_r2)
            all_results['Mean Train Correlation'].append(mean_train_correlation)
            all_results['Std Train Correlation'].append(std_train_correlation)
            all_results['Mean Validation R2'].append(mean_val_r2)
            all_results['Std Validation R2'].append(std_val_r2)
            all_results['Mean Validation Correlation'].append(mean_val_correlation)
            all_results['Std Validation Correlation'].append(std_val_correlation)

            # Report the final statistics for the current target
            logging.info("\nCross-validation process complete!")
            logging.info(f"Mean Test R²: {mean_test_r2:.4f} ± {std_test_r2:.4f}")
            logging.info(f"Mean Test Correlation: {mean_test_correlation:.4f} ± {std_test_correlation:.4f}")
            logging.info(f"Mean Train R²: {mean_train_r2:.4f} ± {std_train_r2:.4f}")
            logging.info(f"Mean Train Correlation: {mean_train_correlation:.4f} ± {std_train_correlation:.4f}")
            logging.info(f"Mean Validation R²: {mean_val_r2:.4f} ± {std_val_r2:.4f}")
            logging.info(f"Mean Validation Correlation: {mean_val_correlation:.4f} ± {std_val_correlation:.4f}\n")

# **Convert the results dictionary to a DataFrame**
df_results = pd.DataFrame(all_results)

# Convert the feature_importance dictionary to a DataFrame
df_feature_importance = pd.DataFrame(feature_importance)

# Calculate the absolute value of coefficients for importance
df_feature_importance['Abs_Coefficient'] = df_feature_importance['Coefficient'].abs()

# Aggregate by Data Type, Subset Type, Target, and Feature
df_feature_agg = df_feature_importance.groupby(['Data Type', 'Subset Type', 'Target', 'Feature']).agg(
    Mean_Coefficient=('Coefficient', 'mean'),
    Mean_Abs_Coefficient=('Abs_Coefficient', 'mean'),
    Std_Abs_Coefficient=('Abs_Coefficient', 'std')
).reset_index()

# Sort features by Mean Absolute Coefficient for each group
df_feature_agg = df_feature_agg.sort_values(
    by=['Data Type', 'Subset Type', 'Target', 'Mean_Abs_Coefficient'],
    ascending=[True, True, True, False]
)

# Save aggregated feature importance to a CSV file
output_dir = 'cross_validation_results'
feature_importance_dir = os.path.join(output_dir, 'feature_importance')
os.makedirs(feature_importance_dir, exist_ok=True)
feature_importance_csv = os.path.join(feature_importance_dir, 'aggregated_feature_importance.csv')
df_feature_agg.to_csv(feature_importance_csv, index=False)
logging.info(f"Aggregated feature importance saved to {feature_importance_csv}")

# **Save the results to a CSV file**
csv_path = os.path.join(output_dir, 'cross_validation_results.csv')
df_results.to_csv(csv_path, index=False)
logging.info(f"Results saved to {csv_path}")
