import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import pandas as pd

def retrieveDataset(data, subset):
    
    if data=='SMRI':
        cognition = pd.read_csv('/data/users4/sdeshpande8/ROI_Mean_Analysis/ROI_Mean_Data/SMRI_Mean_ROI_Cognition.csv')
        cbcl = pd.read_csv('/data/users4/sdeshpande8/ROI_Mean_Analysis/ROI_Mean_Data/SMRI_Mean_ROI_CBCL.csv')
        pca1 = pd.read_csv('/data/users4/sdeshpande8/ROI_Mean_Analysis/ROI_Mean_Data/SMRI_Mean_ROI_PCA1.csv')

    elif data=='DTI':
        cognition = pd.read_csv('/data/users4/sdeshpande8/ROI_Mean_Analysis/ROI_Mean_Data/DTI_Mean_ROI_Cognition.csv')
        cbcl = pd.read_csv('/data/users4/sdeshpande8/ROI_Mean_Analysis/ROI_Mean_Data/DTI_Mean_ROI_CBCL.csv')
        pca1 = pd.read_csv('/data/users4/sdeshpande8/ROI_Mean_Analysis/ROI_Mean_Data/DTI_Mean_ROI_PCA1.csv')
        
    if subset=='FTC':
        cognition = cognition[cognition.columns[~cognition.columns.str.startswith('P')]]
        cbcl = cbcl[cbcl.columns[~cbcl.columns.str.startswith('P')]]
        pca1 = pca1[pca1.columns[~pca1.columns.str.startswith('P')]]
        
    elif subset=='FTP':
        cognition = cognition[cognition.columns[~cognition.columns.str.startswith('C')]]
        cbcl = cbcl[cbcl.columns[~cbcl.columns.str.startswith('C')]]
        pca1 = pca1[pca1.columns[~pca1.columns.str.startswith('C')]]

    ## Getting the Y values
    y3_attention = cognition['tfmri_nb_all_beh_c0b_rate']
    y4_working_memory = cognition['tfmri_nb_all_beh_c2b_rate']
    y_cbcl = cbcl['cbcl_scr_syn_attention_r']
    y_pca1 = pca1['Ave.Standarized_inAttention']

    ## Getting X Values
    X_cog = cognition.drop(columns=['src_subject_id','tfmri_nb_all_beh_c0b_mrt', 'tfmri_nb_all_beh_c2b_stdrt', 'tfmri_nb_all_beh_c0b_rate', 'tfmri_nb_all_beh_c2b_rate'])
    X_cbcl = cbcl.drop(columns = ['src_subject_id', 'cbcl_scr_syn_attention_r'])
    X_pca1 = pca1.drop(columns = ['src_subject_id', 'Ave.Standarized_inAttention'])

    dataset = {'c0b':[X_cog, y3_attention], 'c2b':[X_cog, y4_working_memory], 'cbcl':[X_cbcl, y_cbcl], 'pca1':[X_pca1, y_pca1]}
    
    return dataset

def svr_train_test_with_permutation_importance(X_data, y_data, C=0.1, epsilon=0.1, random_state=42, n_runs=5):
    # Ensure X_data is a DataFrame
    if not isinstance(X_data, pd.DataFrame):
        X_data = pd.DataFrame(X_data)
    
    # Initialize lists to store metrics across runs
    final_train_r2_list = []
    final_train_corr_list = []
    test_r2_list = []
    test_corr_list = []

    # Run the entire process multiple times to compute standard deviations for test metrics
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs} with random_state={random_state + run}")
        # Scale the features and keep as DataFrame
        scaler_X = StandardScaler()
        X_scaled = pd.DataFrame(scaler_X.fit_transform(X_data), columns=X_data.columns)
        
        # Convert y to pandas Series without scaling
        y = pd.Series(y_data.to_numpy().flatten())
        
        # Bin y into quantiles for stratification
        q = len(np.unique(y))  # Adjust number of quantiles
        y_binned = pd.qcut(y, q=30, duplicates='drop')
        
        # Split into training/validation and test sets
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y_binned, random_state=random_state + run
        )
        
        # Initialize KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state + run)
    
        # Initialize lists to store metrics for each fold
        train_r2_scores = []
        train_corr_scores = []
        val_r2_scores = []
        val_corr_scores = []
        
        # Cross-validation loop
        for train_index, val_index in kf.split(X_trainval):
            X_train, X_val = X_trainval.iloc[train_index], X_trainval.iloc[val_index]
            y_train, y_val = y_trainval.iloc[train_index], y_trainval.iloc[val_index]
            
            # Initialize SVR
            svr = SVR(kernel='rbf', C=C, epsilon=epsilon, cache_size=1000)
            svr.fit(X_train, y_train)
            
            # Predictions
            train_predictions = svr.predict(X_train)
            val_predictions = svr.predict(X_val)
            
            # Calculate R²
            train_r2 = r2_score(y_train, train_predictions)
            val_r2 = r2_score(y_val, val_predictions)
            
            # Calculate Pearson Correlation Coefficient
            train_corr, _ = pearsonr(y_train, train_predictions)
            val_corr, _ = pearsonr(y_val, val_predictions)
            
            # Append metrics
            train_r2_scores.append(train_r2)
            train_corr_scores.append(train_corr)
            val_r2_scores.append(val_r2)
            val_corr_scores.append(val_corr)
        
        # Calculate mean and std of metrics across folds
        mean_train_r2 = np.mean(train_r2_scores)
        std_train_r2 = np.std(train_r2_scores)
        mean_train_corr = np.mean(train_corr_scores)
        std_train_corr = np.std(train_corr_scores)
        
        mean_val_r2 = np.mean(val_r2_scores)
        std_val_r2 = np.std(val_r2_scores)
        mean_val_corr = np.mean(val_corr_scores)
        std_val_corr = np.std(val_corr_scores)
        
        print("--------------------------------------------------------------------------------")
        print(f"Run {run + 1}:")
        print(f"Average Train R²: {mean_train_r2:.4f} ± {std_train_r2:.4f}")
        print(f"Average Train Corr: {mean_train_corr:.4f} ± {std_train_corr:.4f}")
        print(f"Average Val R²: {mean_val_r2:.4f} ± {std_val_r2:.4f}")
        print(f"Average Val Corr: {mean_val_corr:.4f} ± {std_val_corr:.4f}")
        
        # Fit final model on the entire training data
        svr_final = SVR(kernel='rbf', C=C, epsilon=epsilon, cache_size=1000)
        svr_final.fit(X_trainval, y_trainval)
        final_train_predictions = svr_final.predict(X_trainval)
        test_predictions = svr_final.predict(X_test)
        
        # Calculate R² for final train and test sets
        final_train_r2 = r2_score(y_trainval, final_train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        
        # Calculate Pearson Correlation Coefficient for final train and test sets
        final_train_corr, _ = pearsonr(y_trainval, final_train_predictions)
        test_corr, _ = pearsonr(y_test, test_predictions)
        
        print("--------------------------------------------------------------------------------")
        print(f"Final Train R²: {final_train_r2:.4f}")
        print(f"Final Train Corr: {final_train_corr:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test Corr: {test_corr:.4f}")
        
        # Append final metrics to lists
        final_train_r2_list.append(final_train_r2)
        final_train_corr_list.append(final_train_corr)
        test_r2_list.append(test_r2)
        test_corr_list.append(test_corr)
        
    # Calculate mean and std of final train and test metrics across runs
    mean_final_train_r2 = np.mean(final_train_r2_list)
    std_final_train_r2 = np.std(final_train_r2_list)
    mean_final_train_corr = np.mean(final_train_corr_list)
    std_final_train_corr = np.std(final_train_corr_list)
    
    mean_test_r2 = np.mean(test_r2_list)
    std_test_r2 = np.std(test_r2_list)
    mean_test_corr = np.mean(test_corr_list)
    std_test_corr = np.std(test_corr_list)
    
    print("================================================================================")
    print(f"Final Train R²: {mean_final_train_r2:.4f} ± {std_final_train_r2:.4f}")
    print(f"Final Train Corr: {mean_final_train_corr:.4f} ± {std_final_train_corr:.4f}")
    print(f"Test R²: {mean_test_r2:.4f} ± {std_test_r2:.4f}")
    print(f"Test Corr: {mean_test_corr:.4f} ± {std_test_corr:.4f}")
    
    # Since permutation importance can be time-consuming, we'll compute it once
    print("Computing permutation importances on the last run...")
    result = permutation_importance(
        svr_final, X_test, y_test, n_repeats=5, random_state=random_state, scoring='r2', n_jobs=-1
    )
    
    # Extract feature importances
    feature_importances = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': result.importances_mean
    })
    
    # Sort feature importances
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    
    # Print feature importances
    print("--------------------------------------------------------------------------------")
    print("Feature Importances (Permutation Importance):")
    print(feature_importances)
    
    # Return the feature importances DataFrame and metrics
    metrics = {
        'mean_train_r2': mean_train_r2,
        'std_train_r2': std_train_r2,
        'mean_train_corr': mean_train_corr,
        'std_train_corr': std_train_corr,
        'mean_val_r2': mean_val_r2,
        'std_val_r2': std_val_r2,
        'mean_val_corr': mean_val_corr,
        'std_val_corr': std_val_corr,
        'mean_final_train_r2': mean_final_train_r2,
        'std_final_train_r2': std_final_train_r2,
        'mean_final_train_corr': mean_final_train_corr,
        'std_final_train_corr': std_final_train_corr,
        'mean_test_r2': mean_test_r2,
        'std_test_r2': std_test_r2,
        'mean_test_corr': mean_test_corr,
        'std_test_corr': std_test_corr
    }
    
    return feature_importances, metrics

# Parallel processing for outer loop over targets and subsets
from itertools import product

def process_target_subset(target_subset):
    target, subset = target_subset
    dataset = retrieveDataset('SMRI', subset)
    X_data = dataset[target][0]
    y_data = dataset[target][1]
    
    print(f'Calculating for {target} in {subset}-------------------------------------------------------------------------')
    feature_imp, metrics = svr_train_test_with_permutation_importance(X_data, y_data, C = 0.03, n_runs=5)
    feature_imp.to_csv(f'SVR_Results5/SMRI_{target}_{subset}_importances.csv', index=False)
    
    # Save metrics to a CSV file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'SVR_Results5/SMRI_{target}_{subset}_metrics.csv', index=False)

# List of targets and subsets
targets = ['c0b', 'c2b']
subsets = ['FTPC', 'FTP', 'FTC']

# Create a list of all combinations of targets and subsets
target_subset_list = list(product(targets, subsets))

# Use joblib to parallelize the outer loop
Parallel(n_jobs=-1)(
    delayed(process_target_subset)(target_subset) for target_subset in target_subset_list
)
