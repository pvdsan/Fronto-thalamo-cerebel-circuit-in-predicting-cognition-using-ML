import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from common_utils import (
    normalize_data,
    train_one_epoch,
    validate_one_epoch2,
    retrieveDataset,
    ensure_directory_exists,
)

class RegressionNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x


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
    
    
    print(X_cog.shape)
    print(X_cbcl.shape)
    print(X_pca1.shape)

    dataset = {'c0b':[X_cog, y3_attention], 'c2b':[X_cog, y4_working_memory], 'cbcl':[X_cbcl, y_cbcl], 'pca1':[X_pca1, y_pca1]}
    
    return dataset




def simple_cross_validation(dataType='SMRI', target='c0b', subsets=['FTC', 'FTP', 'FTPC'], num_folds=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dir_name = f'neural_net_results_{dataType}_{target}'
    ensure_directory_exists(dir_name)

    # Fixed hyperparameters
    hyperparameters = {
        'lr': 1e-5,
        'batch_size': 64,
        'num_epochs': 750
    }

    for subset in subsets:
        print(f'Cross-Validation for target {target} and subset {subset}')
        dataset = retrieveDataset(data=dataType, subset=subset)
        if target not in dataset:
            print(f"Target {target} not found in dataset for subset {subset}")
            continue
        X, y = dataset[target]
        X, y = normalize_data(X=X, y=y, by='std')
        featureLen = X.shape[1]
        
        # Split into training+validation and test sets
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create KFold object
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
            print(f'Fold {fold + 1}/{num_folds}')
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
            
            # Create DataLoaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)
            
            # Initialize model
            model = RegressionNet(featureLen).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'])
            
                        # Initialize LR Scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            
            num_epochs = hyperparameters['num_epochs']
            
            # Lists to store metrics
            train_losses = []
            train_r2_scores = []
            val_losses = []
            val_r2_scores = []
            val_corr_scores = []
            
            # Train the model
            for epoch in range(num_epochs):
                train_loss, train_r2 = train_one_epoch(model, train_loader, criterion, optimizer, device)
                train_losses.append(train_loss)
                train_r2_scores.append(train_r2)
                
                # Validate the model
                val_loss, val_r2, val_corr = validate_one_epoch2(model, val_loader, criterion, device)
                val_losses.append(val_loss)
                val_r2_scores.append(val_r2)
                val_corr_scores.append(val_corr)
                
                scheduler.step(val_loss)
                
                print(f'Epoch {epoch+1}/{num_epochs} - '
                      f'Train Loss: {train_loss:.4f}, Train R²: {train_r2:.4f} - '
                      f'Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}')
            
            # Store results for this fold
            fold_results.append({
                'fold': fold + 1,
                'train_r2': train_r2_scores[-1],  # Last epoch
                'train_corr': val_corr_scores[-1],
                'val_r2': val_r2_scores[-1],
                'val_corr': val_corr_scores[-1],
            })
            
            print(f'Fold {fold + 1} Results:')
            print(f'Train R²: {train_r2_scores[-1]}, Train Corr: {val_corr_scores[-1]}')
            print(f'Val R²: {val_r2_scores[-1]}, Val Corr: {val_corr_scores[-1]}')
            print('--------------------------------------------------------')
            
            # Save epoch-wise metrics for this fold
            metrics_df = pd.DataFrame({
                'epoch': np.arange(1, num_epochs + 1),
                'train_loss': train_losses,
                'train_r2': train_r2_scores,
                'val_loss': val_losses,
                'val_r2': val_r2_scores,
                'val_corr': val_corr_scores,
            })
            metrics_file_path = os.path.join(
                dir_name, f'{dataType}_{target}_{subset}_fold{fold+1}_metrics.csv'
            )
            metrics_df.to_csv(metrics_file_path, index=False)
            print(f'Epoch-wise metrics for fold {fold + 1} saved to {metrics_file_path}')
        
        # After cross-validation, train on entire training+validation set
        train_dataset_full = TensorDataset(X_trainval, y_trainval)
        train_loader_full = DataLoader(train_dataset_full, batch_size=hyperparameters['batch_size'], shuffle=True)
        
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)
        
        # Initialize model
        model = RegressionNet(featureLen).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'])
        
        
                # Initialize LR Scheduler for full training
        scheduler_full = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        num_epochs = hyperparameters['num_epochs']
        
        # Lists to store metrics for full training
        train_losses_full = []
        train_r2_scores_full = []
        test_losses_full = []
        test_r2_scores_full = []
        test_corr_scores_full = []
        
        # Train the model on full training data
        for epoch in range(num_epochs):
            train_loss, train_r2 = train_one_epoch(model, train_loader_full, criterion, optimizer, device)
            train_losses_full.append(train_loss)
            train_r2_scores_full.append(train_r2)
            
            # Optionally, evaluate on test data at each epoch
            test_loss, test_r2, test_corr = validate_one_epoch2(model, test_loader, criterion, device)
            test_losses_full.append(test_loss)
            test_r2_scores_full.append(test_r2)
            test_corr_scores_full.append(test_corr)
            
            scheduler_full.step(test_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs} - '
                  f'Train Loss: {train_loss:.4f}, Train R²: {train_r2:.4f} - '
                  f'Test Loss: {test_loss:.4f}, Test R²: {test_r2:.4f}')
        
        # Evaluate on final training data
        train_loss, train_r2, train_corr = validate_one_epoch2(model, train_loader_full, criterion, device)
        
        # Evaluate on test data
        test_loss, test_r2, test_corr = validate_one_epoch2(model, test_loader, criterion, device)
        
        # Save overall results
        overall_results = {
            'train_r2': train_r2,
            'train_corr': train_corr,
            'test_r2': test_r2,
            'test_corr': test_corr,
        }
        
        print('Final Model Results on Test Set:')
        print(f'Train R²: {train_r2}, Train Corr: {train_corr}')
        print(f'Test R²: {test_r2}, Test Corr: {test_corr}')
        print('--------------------------------------------------------')
        
        # Save cross-validation results
        results_df = pd.DataFrame(fold_results)
        file_path = os.path.join(dir_name, f'{dataType}_{target}_{subset}_cv_results.csv')
        results_df.to_csv(file_path, index=False)
        print(f'Cross-validation results saved to {file_path}')
        
        # Save test set results
        overall_results_df = pd.DataFrame([overall_results])
        overall_file_path = os.path.join(dir_name, f'{dataType}_{target}_{subset}_test_results.csv')
        overall_results_df.to_csv(overall_file_path, index=False)
        print(f'Test results saved to {overall_file_path}')
        
        # Save epoch-wise metrics for full training
        metrics_full_df = pd.DataFrame({
            'epoch': np.arange(1, num_epochs + 1),
            'train_loss': train_losses_full,
            'train_r2': train_r2_scores_full,
            'test_loss': test_losses_full,
            'test_r2': test_r2_scores_full,
            'test_corr': test_corr_scores_full,
        })
        metrics_full_file_path = os.path.join(
            dir_name, f'{dataType}_{target}_{subset}_full_training_metrics.csv'
        )
        metrics_full_df.to_csv(metrics_full_file_path, index=False)
        print(f'Epoch-wise metrics for full training saved to {metrics_full_file_path}')

# Run the cross-validation for a specific target and multiple subsets
simple_cross_validation(target='c0b', subsets=['FTC', 'FTP', 'FTPC'])
