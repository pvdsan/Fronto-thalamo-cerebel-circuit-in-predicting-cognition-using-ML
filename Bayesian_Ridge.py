from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import os
import matplotlib.pyplot as plt
from common_utils import retrieveDataset, normalize_data
from tqdm import tqdm


data_types = ['SMRI', 'DTI']
subset_types = ['FTPC', 'FTP', 'FTC']


for data_type in data_types:
    
    print(f" Starting training for {data_type} ")
    for subset_type in subset_types:
        
        print(f"Training for subset{subset_type}")
        # Retrieve and normalize dataset
        dataset = retrieveDataset(data=data_type, subset=subset_type)
    
        for target, (X, y) in dataset.items():
        
            print(f" Target:{target}")
            X, y = normalize_data(X, y)
            # Convert to numpy arrays of float32 for efficiency
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32).ravel()

            # Define the outer 5-fold cross-validation
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

            # Hyperparameters for Bayesian Ridge Regression
            param_grid = {
                'alpha_1': [1e-6, 1e-5, 1e-4],
                'alpha_2': [1e-6, 1e-5, 1e-4],
                'lambda_1': [1e-6, 1e-5, 1e-4],
                'lambda_2': [1e-6, 1e-5, 1e-4]
            }

            # Store test scores, R², and correlations from outer folds
            test_scores = []
            train_r2_scores = []
            test_r2_scores = []
            train_correlations = []
            test_correlations = []

            validation_r2_scores = []
            validation_correlations = []

            print("Starting nested cross-validation process...\n")

            # Outer loop of 5-fold cross-validation with tqdm progress bar
            for outer_fold, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(X), total=5, desc="Outer CV Folds"), 1):
                print(f"\nProcessing outer fold {outer_fold}...")

                # Split the data into training and testing sets
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Inner loop: Perform hyperparameter tuning with 5-fold cross-validation
                inner_cv = KFold(n_splits=5, shuffle=True)
                model = BayesianRidge()
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='r2', n_jobs=16, return_train_score=True)

                print(f"Starting hyperparameter tuning for outer fold {outer_fold}...")
                # Fit the model with the best hyperparameters on the training data
                grid_search.fit(X_train, y_train)

                # Use the best estimator from inner cross-validation
                best_model = grid_search.best_estimator_

                # Print the best hyperparameters for the current outer fold
                print(f"Best hyperparameters for outer fold {outer_fold}: {grid_search.best_params_}")

                # Evaluate the best model on the outer test data
                y_pred_train = best_model.predict(X_train)
                y_pred_test = best_model.predict(X_test)

                # Calculate R² scores for outer fold (train and test)
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)

                # Calculate mean squared error for test fold
                test_score = mean_squared_error(y_test, y_pred_test)

                # Calculate Pearson correlation for train and test data
                train_correlation, _ = pearsonr(y_train, y_pred_train)
                test_correlation, _ = pearsonr(y_test, y_pred_test)

                # Store scores and correlations for train and test
                test_scores.append(test_score)
                train_r2_scores.append(train_r2)
                test_r2_scores.append(test_r2)
                train_correlations.append(train_correlation)
                test_correlations.append(test_correlation)

                # Calculate validation predictions using cross_val_predict
                y_val_pred = cross_val_predict(best_model, X_train, y_train, cv=inner_cv)
                
                # Calculate R² and correlation for validation
                val_r2 = r2_score(y_train, y_val_pred)
                validation_r2_scores.append(val_r2)
                
                val_correlation, _ = pearsonr(y_train, y_val_pred)
                validation_correlations.append(val_correlation)

                print(f"Completed outer fold {outer_fold}. Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Test MSE: {test_score:.4f}, Train Correlation: {train_correlation:.4f}, Test Correlation: {test_correlation:.4f}, Validation R²: {val_r2:.4f}")

            # Calculate mean and std for MSE, R², and correlations
            mean_test_score = np.mean(test_scores)
            std_test_score = np.std(test_scores)
            mean_train_r2 = np.mean(train_r2_scores)
            std_train_r2 = np.std(train_r2_scores)
            mean_test_r2 = np.mean(test_r2_scores)
            std_test_r2 = np.std(test_r2_scores)
            mean_train_correlation = np.mean(train_correlations)
            std_train_correlation = np.std(train_correlations)
            mean_test_correlation = np.mean(test_correlations)
            std_test_correlation = np.std(test_correlations)

            mean_val_r2 = np.mean(validation_r2_scores)
            std_val_r2 = np.std(validation_r2_scores)
            mean_val_correlation = np.mean(validation_correlations)
            std_val_correlation = np.std(validation_correlations)

            # Report the final statistics
            print("\nCross-validation process complete!")
            print(f"Mean test score (MSE): {mean_test_score:.4f}, Std: {std_test_score:.4f}")
            print(f"Mean train R²: {mean_train_r2:.4f}, Std: {std_train_r2:.4f}")
            print(f"Mean test R²: {mean_test_r2:.4f}, Std: {std_test_r2:.4f}")
            print(f"Mean train correlation: {mean_train_correlation:.4f}, Std: {std_train_correlation:.4f}")
            print(f"Mean test correlation: {mean_test_correlation:.4f}, Std: {std_test_correlation:.4f}")
            print(f"Mean validation R²: {mean_val_r2:.4f}, Std: {std_val_r2:.4f}")
            print(f"Mean validation correlation: {mean_val_correlation:.4f}, Std: {std_val_correlation:.4f}")