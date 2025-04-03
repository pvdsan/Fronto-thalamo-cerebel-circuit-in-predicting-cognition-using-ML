import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from RegressionNet import RegressionNet
import pandas as pd
from sklearn.model_selection import KFold
from itertools import cycle
from common_utils import (
    normalize_data,
    training_epoch,
    validation_epoch,
    retrieveDataset,
    ensure_directory_exists,
)
import logging
import torch.multiprocessing as mp
import glob

def process_outer_fold(args):
    (
        dataType,
        subset,
        target,
        outer_fold,
        num_outer_folds,
        num_inner_folds,
        num_epochs,
        device_id,
        dir_name,
        hyperparameter_list,
        batch_size
    ) = args

    # Set device
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available()
                          else 'cpu')

    # Configure logging per process
    logger = logging.getLogger(
        f'{dataType}_{target}_{subset}_outer_fold_{outer_fold + 1}')
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    log_filename = os.path.join(
        dir_name,
        f'{dataType}_{target}_{subset}_process_outer_fold_{outer_fold + 1}.log'
    )
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the handlers to the logger
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)

    logger.info('Starting process for outer fold...')

    # Load data
    dataset = retrieveDataset(data=dataType, subset=subset)
    if target not in dataset:
        logger.warning(f"Target {target} not found in dataset for subset "
                       f"{subset}")
        return

    X_df, y_df = dataset[target]
    feature_names = X_df.columns.tolist()
    X, y = normalize_data(X=X_df, y=y_df, by='std', normalize_y=False)
    featureLen = X.shape[1]

    # Create outer KFold object
    outer_kf = KFold(n_splits=num_outer_folds, shuffle=True, random_state=42)

    # Get the trainval_idx and test_idx for the given outer_fold
    for fold_index, (trainval_idx, test_idx) in enumerate(outer_kf.split(X)):
        if fold_index == outer_fold:
            # Proceed with this fold
            break
    else:
        logger.error(f"Outer fold {outer_fold} not found.")
        return

    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]

    # Initialize feature importance dictionary
    feature_importance = {
        'Data Type': [],
        'Subset Type': [],
        'Target': [],
        'Outer Fold': [],
        'Feature': [],
        'Mean Gradient': [],
        'Std Gradient': []
    }

    # Create inner KFold object
    inner_kf = KFold(n_splits=num_inner_folds, shuffle=True,
                     random_state=outer_fold + 42)

    inner_fold_results = []
    hyperparam_cycle = cycle(hyperparameter_list)  # To assign hyperparameters

    best_val_r2_outer = -np.inf
    best_lr_outer = 0
    
    epochs_no_improve = 0
    early_stop_patience = 25  # Stop if no improvement in val_r2 for 25 epochs

    for inner_fold, (train_idx, val_idx) in enumerate(
            inner_kf.split(X_trainval)):
        hyperparams = next(hyperparam_cycle)
        lr = hyperparams['lr']
        logger.info(f'  Inner Fold {inner_fold + 1}/{num_inner_folds} with '
                    f'hyperparameters: {hyperparams}')

        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, pin_memory=True)

        # Initialize model
        model = RegressionNet(featureLen).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Initialize LR Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.50, patience=8)

        # Variables to track the best validation performance
        best_val_r2 = -np.inf
        best_epoch = -1
        best_metrics = {}

        # Train the model
        for epoch in range(num_epochs):
            train_loss, train_r2, train_corr = training_epoch(
                model, train_loader, criterion, optimizer, device)

            # Validate the model
            val_loss, val_r2, val_corr = validation_epoch(
                model, val_loader, criterion, device)

            scheduler.step(val_loss)

            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, '
                            f'Val Loss: {val_loss:.4f}, Train R2: '
                            f'{train_r2:.4f}, Val R2: {val_r2:.4f}')

            # Check if current validation R² is the best
            if val_r2 > best_val_r2:
                logger.info(f'Best val R² improved from {best_val_r2:.4f} to '
                            f'{val_r2:.4f}')
                best_val_r2 = val_r2
                best_epoch = epoch + 1
                best_metrics = {
                    'train_loss': train_loss,
                    'train_r2': train_r2,
                    'train_corr': train_corr,
                    'val_loss': val_loss,
                    'val_r2': val_r2,
                    'val_corr': val_corr
                }
                epochs_no_improve = 0  # Reset counter
                
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    logger.info(f'Early stopping at epoch {epoch+1} after '
                                f'{epochs_no_improve} epochs with no improvement in val R².')
                    break

        # Store results for this inner fold
        inner_fold_results.append({
            'outer_fold': outer_fold + 1,
            'inner_fold': inner_fold + 1,
            'hyperparameters': hyperparams,
            'best_epoch': best_epoch,
            'train_r2': best_metrics['train_r2'],
            'train_corr': best_metrics['train_corr'],
            'val_r2': best_metrics['val_r2'],
            'val_corr': best_metrics['val_corr'],
        })

        logger.info(f'Inner Fold {inner_fold + 1} Results:')
        logger.info(f'Hyperparameters: {hyperparams}')
        logger.info(f'Best Epoch: {best_epoch}')
        logger.info(f'Train R²: {best_metrics["train_r2"]:.4f}, Train Corr: '
                    f'{best_metrics["train_corr"]:.4f}')
        logger.info(f'Val R²: {best_metrics["val_r2"]:.4f}, Val Corr: '
                    f'{best_metrics["val_corr"]:.4f}')
        logger.info('--------------------------------------------------------')

        if best_metrics["val_r2"] > best_val_r2_outer:
            best_val_r2_outer = best_val_r2
            best_lr_outer = lr

    # Compute aggregate val_r2 and val_corr across inner folds
    val_r2_list = [fold_result['val_r2'] for fold_result in inner_fold_results]
    val_corr_list = [fold_result['val_corr'] for fold_result in
                     inner_fold_results]

    val_r2_mean = np.mean(val_r2_list)
    val_r2_std = np.std(val_r2_list)
    val_corr_mean = np.mean(val_corr_list)
    val_corr_std = np.std(val_corr_list)

    logger.info(f'Aggregate Inner Fold Results for Outer Fold '
                f'{outer_fold + 1}:')
    logger.info(f'Val R² Mean: {val_r2_mean:.4f}, Val R² Std: '
                f'{val_r2_std:.4f}')
    logger.info(f'Val Corr Mean: {val_corr_mean:.4f}, Val Corr Std: '
                f'{val_corr_std:.4f}')

    logger.info(f'Training with the best lr: {best_lr_outer}')

    # Train on full training data with selected hyperparameters
    train_dataset_full = TensorDataset(X_trainval, y_trainval)
    train_loader_full = DataLoader(train_dataset_full, batch_size=batch_size,
                                   shuffle=True, pin_memory=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, pin_memory=True)

    # Initialize model
    model = RegressionNet(featureLen).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_lr_outer)

    # Initialize LR Scheduler for full training
    scheduler_full = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.50, patience=8)

    # Variables to track the best test performance
    best_test_r2 = -np.inf
    best_epoch_full = -1
    best_metrics_full = {}
    num_epochs_run_full = 0  # Actual number of epochs run

    # Train the model on full training data
    for epoch in range(num_epochs):
        num_epochs_run_full += 1
        train_loss, train_r2, train_corr = training_epoch(
            model, train_loader_full, criterion, optimizer, device)

        # Evaluate on test data
        test_loss, test_r2, test_corr = validation_epoch(
            model, test_loader, criterion, device)

        scheduler_full.step(test_loss)

        if epoch % 10 == 0:
            logger.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, '
                        f'Val Loss: {test_loss:.4f}, Train R2: '
                        f'{train_r2:.4f}, Val R2: {test_r2:.4f}')

        # Check if current test R² is the best
        if test_r2 > best_test_r2:
            logger.info(f'Best val R² improved from {best_test_r2:.4f} to '
                        f'{test_r2:.4f}')
            best_test_r2 = test_r2
            best_epoch_full = epoch + 1
            best_metrics_full = {
                'train_loss': train_loss,
                'train_r2': train_r2,
                'train_corr': train_corr,
                'test_loss': test_loss,
                'test_r2': test_r2,
                'test_corr': test_corr
            }
            epochs_no_improve = 0  # Reset counter
            
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info(f'Early stopping at epoch {epoch+1} after '
                            f'{epochs_no_improve} epochs with no improvement in val R².')
                break

    # **Compute Gradient-Based Feature Importance**
    logger.info(f'Performing gradient-based feature importance analysis for Outer Fold {outer_fold + 1}')

    # Ensure model is in evaluation mode
    model.eval()

    # Select a subset of test data for gradient analysis
    num_samples = min(1000, len(X_test))
    X_test_subset = X_test[:num_samples].to(device)
    y_test_subset = y_test[:num_samples].to(device)

    # Compute gradients of outputs w.r.t inputs
    batch_size_grad = 64  # Adjust based on your system's memory
    gradients_list = []

    for i in range(0, len(X_test_subset), batch_size_grad):
        X_batch = X_test_subset[i:i + batch_size_grad]
        X_batch.requires_grad = True
        outputs = model(X_batch)  # Shape: [batch_size, 1]

        # Create gradient outputs (ones)
        grad_outputs = torch.ones_like(outputs)

        # Compute gradients w.r.t inputs
        outputs.backward(grad_outputs)

        # Get gradients
        gradients = X_batch.grad.detach().cpu().numpy()  # Shape: [batch_size, num_features]
        gradients_list.append(gradients)

    # Concatenate all gradients
    gradients_np = np.concatenate(gradients_list, axis=0)  # Shape: [num_samples, num_features]

    # Compute mean and std of gradients per feature
    mean_gradients = gradients_np.mean(axis=0)
    std_gradients = gradients_np.std(axis=0)

    # Store gradient values per feature
    for feature, mean_val, std_val in zip(feature_names, mean_gradients, std_gradients):
        feature_importance['Data Type'].append(dataType)
        feature_importance['Subset Type'].append(subset)
        feature_importance['Target'].append(target)
        feature_importance['Outer Fold'].append(outer_fold + 1)
        feature_importance['Feature'].append(feature)
        feature_importance['Mean Gradient'].append(mean_val)
        feature_importance['Std Gradient'].append(std_val)

    logger.info(f'Gradient-based feature importance analysis completed for Outer Fold {outer_fold + 1}')

    # Save overall results for this outer fold
    outer_fold_results = [{
        'outer_fold': outer_fold + 1,
        'hyperparameters': best_lr_outer,
        'best_epoch': best_epoch_full,
        'train_r2': best_metrics_full['train_r2'],
        'train_corr': best_metrics_full['train_corr'],
        'test_r2': best_metrics_full['test_r2'],
        'test_corr': best_metrics_full['test_corr'],
        # Include aggregate validation metrics
        'val_r2_mean': val_r2_mean,
        'val_r2_std': val_r2_std,
        'val_corr_mean': val_corr_mean,
        'val_corr_std': val_corr_std,
    }]

    logger.info(f'Outer Fold {outer_fold + 1} Results on Test Set:')
    logger.info(f'Hyperparameters: {best_lr_outer}')
    logger.info(f'Best Epoch: {best_epoch_full}')
    logger.info(f'Train R²: {best_metrics_full["train_r2"]:.4f}, Train Corr: '
                f'{best_metrics_full["train_corr"]:.4f}')
    logger.info(f'Test R²: {best_metrics_full["test_r2"]:.4f}, Test Corr: '
                f'{best_metrics_full["test_corr"]:.4f}')
    logger.info('--------------------------------------------------------')

    # Save inner fold results for this outer fold
    inner_results_df = pd.DataFrame(inner_fold_results)
    inner_results_file_path = os.path.join(
        dir_name,
        f'{dataType}_{target}_{subset}_outerfold{outer_fold+1}_inner_results.csv'
    )
    inner_results_df.to_csv(inner_results_file_path, index=False)
    logger.info(f'Inner fold results saved to {inner_results_file_path}')

    # Save outer fold results
    outer_results_df = pd.DataFrame(outer_fold_results)
    outer_results_file_path = os.path.join(
        dir_name,
        f'{dataType}_{target}_{subset}_outerfold{outer_fold+1}_outer_cv_results.csv'
    )
    outer_results_df.to_csv(outer_results_file_path, index=False)
    logger.info(f'Outer cross-validation results saved to '
                f'{outer_results_file_path}')

    # Save feature importance
    feature_importance_df = pd.DataFrame(feature_importance)
    feature_importance_file = os.path.join(
        dir_name,
        f'{dataType}_{target}_{subset}_outerfold{outer_fold+1}_feature_importance.csv'
    )
    feature_importance_df.to_csv(feature_importance_file, index=False)
    logger.info(f'Feature importance saved to {feature_importance_file}')

def nested_cross_validation(dataTypes=['SMRI', 'DTI'], target='c0b',
                            subsets=['FTPC', 'FTC', 'FTP'], num_outer_folds=5,
                            num_inner_folds=5, num_epochs=1000, num_gpus=4):
    # The hyperparameters and batch size can be adjusted as needed
    hyperparameter_list = [  
        {'lr': 1.2e-5, },
        {'lr': 1.4e-5, },
        {'lr': 1.6e-5, },
        {'lr': 1.8e-5, },
        {'lr': 2e-5, }
    ]
    batch_size = 128

    for dataType in dataTypes:
        dir_name = f'neural_net_nested_cv_results_Gradient_{dataType}_{target}'
        ensure_directory_exists(dir_name)

        for subset in subsets:
            logging.info(f'Nested Cross-Validation for target {target} and '
                         f'subset {subset}')

            dataset = retrieveDataset(data=dataType, subset=subset)
            if target not in dataset:
                logging.warning(f"Target {target} not found in dataset for "
                                f"subset {subset}")
                continue

            # Create a list of arguments for each outer fold
            args_list = []
            for outer_fold in range(num_outer_folds):
                device_id = outer_fold % num_gpus
                args = (
                    dataType,
                    subset,
                    target,
                    outer_fold,
                    num_outer_folds,
                    num_inner_folds,
                    num_epochs,
                    device_id,
                    dir_name,
                    hyperparameter_list,
                    batch_size
                )
                args_list.append(args)

            # Use multiprocessing Pool to run outer folds in parallel
            with mp.Pool(processes=num_gpus) as pool:
                pool.map(process_outer_fold, args_list)

            # After all processes have completed, aggregate the results
            # Aggregate outer fold results
            outer_results_files = [
                os.path.join(
                    dir_name,
                    f'{dataType}_{target}_{subset}_outerfold{fold+1}_outer_cv_results.csv'
                ) for fold in range(num_outer_folds)
            ]
            outer_results_df = pd.concat([pd.read_csv(f) for f in
                                          outer_results_files])
            outer_results_file_path = os.path.join(
                dir_name,
                f'{dataType}_{target}_{subset}_outer_cv_results.csv')
            outer_results_df.to_csv(outer_results_file_path, index=False)
            logging.info(f'Aggregated outer cross-validation results saved to '
                         f'{outer_results_file_path}')

            # Optionally, compute overall aggregates across outer folds
            overall_metrics = {
                'test_r2_mean': outer_results_df['test_r2'].mean(),
                'test_r2_std': outer_results_df['test_r2'].std(),
                'test_corr_mean': outer_results_df['test_corr'].mean(),
                'test_corr_std': outer_results_df['test_corr'].std(),
                'val_r2_mean': outer_results_df['val_r2_mean'].mean(),
                'val_r2_std': outer_results_df['val_r2_mean'].std(),
                'val_corr_mean': outer_results_df['val_corr_mean'].mean(),
                'val_corr_std': outer_results_df['val_corr_mean'].std(),
            }

            logging.info('Overall Performance Metrics Across Outer Folds:')
            for metric_name, value in overall_metrics.items():
                logging.info(f'{metric_name}: {value:.4f}')

            # Aggregate feature importance
            feature_importance_files = [
                os.path.join(
                    dir_name,
                    f'{dataType}_{target}_{subset}_outerfold{fold+1}_feature_importance.csv'
                ) for fold in range(num_outer_folds)
            ]
            df_feature_importance = pd.concat([pd.read_csv(f) for f in
                                               feature_importance_files])

            # **Gradient-Based Feature Importance Aggregation and Saving**
            # Aggregate gradients by Data Type, Subset Type, Target, and Feature
            df_feature_agg = df_feature_importance.groupby(
                ['Data Type', 'Subset Type', 'Target', 'Feature']
            ).agg(
                Mean_Gradient=('Mean Gradient', 'mean'),
                Std_Gradient=('Std Gradient', 'mean')  # Mean of std deviations
            ).reset_index()

            # Sort features by absolute Mean Gradient for each group
            df_feature_agg['Abs_Mean_Gradient'] = df_feature_agg['Mean_Gradient'].abs()
            df_feature_agg = df_feature_agg.sort_values(
                by=['Data Type', 'Subset Type', 'Target', 'Abs_Mean_Gradient'],
                ascending=[True, True, True, False]
            )

            # Save aggregated feature importance to a CSV file
            feature_importance_dir = os.path.join(dir_name, 'feature_importance')
            os.makedirs(feature_importance_dir, exist_ok=True)
            feature_importance_csv = os.path.join(
                feature_importance_dir,
                f'{dataType}_{target}_{subset}_aggregated_feature_importance_gradient.csv'
            )
            df_feature_agg.to_csv(feature_importance_csv, index=False)
            logging.info(f"Aggregated gradient-based feature importance saved to "
                         f"{feature_importance_csv}")

    # Optionally, aggregate feature importance across all subsets
    all_feature_importance_files = glob.glob(
        os.path.join(dir_name, 'feature_importance',
                     f'{dataType}_{target}_*_aggregated_feature_importance_gradient.csv')
    )
    df_all_feature_importance = pd.concat(
        [pd.read_csv(f) for f in all_feature_importance_files])

    # Aggregate across all subsets
    df_overall_feature_agg = df_all_feature_importance.groupby(
        ['Data Type', 'Target', 'Feature']
    ).agg(
        Mean_Gradient=('Mean_Gradient', 'mean'),
        Std_Gradient=('Std_Gradient', 'mean')
    ).reset_index()

    # Save overall aggregated feature importance
    overall_feature_importance_csv = os.path.join(
        dir_name, 'feature_importance',
        f'{dataType}_{target}_overall_aggregated_feature_importance_gradient.csv')
    df_overall_feature_agg.to_csv(overall_feature_importance_csv, index=False)
    logging.info(f"Overall aggregated gradient-based feature importance saved to "
                 f"{overall_feature_importance_csv}")

if __name__ == '__main__':
    # Set the multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Run the nested cross-validation using 4 GPUs
    nested_cross_validation(
        dataTypes=['SMRI'],
        target='c0b',
        subsets=['FTP', 'FTC'],
        num_gpus=4
    )
