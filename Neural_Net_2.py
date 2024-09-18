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
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x

def simple_cross_validation(dataType='SMRI', subset='FTPC', num_folds=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = retrieveDataset(data=dataType, subset=subset)
    dir_name = f'neural_net_results_{dataType}_{subset}'
    ensure_directory_exists(dir_name)
    
    # Fixed hyperparameters
    hyperparameters = {
        'lr': 1e-4,
        'batch_size': 64,
        'num_epochs': 50
    }
    
    for target, (X, y) in dataset.items():
        print(f'Cross-Validation for {target}')
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
            
            num_epochs = hyperparameters['num_epochs']
            
            # Train the model
            for epoch in range(num_epochs):
                train_loss, train_r2 = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate the model
            val_loss, val_r2, val_corr = validate_one_epoch2(model, val_loader, criterion, device)
            
            # Evaluate on training data
            train_loss, train_r2, train_corr = validate_one_epoch2(model, train_loader, criterion, device)
            
            # Store results
            fold_results.append({
                'fold': fold + 1,
                'train_r2': train_r2,
                'train_corr': train_corr,
                'val_r2': val_r2,
                'val_corr': val_corr,
            })
            
            print(f'Fold {fold + 1} Results:')
            print(f'Train R²: {train_r2}, Train Corr: {train_corr}')
            print(f'Val R²: {val_r2}, Val Corr: {val_corr}')
            print('--------------------------------------------------------')
        
        # After cross-validation, train on entire training+validation set
        train_dataset_full = TensorDataset(X_trainval, y_trainval)
        train_loader_full = DataLoader(train_dataset_full, batch_size=hyperparameters['batch_size'], shuffle=True)
        
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)
        
        # Initialize model
        model = RegressionNet(featureLen).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'])
        
        num_epochs = hyperparameters['num_epochs']
        
        # Train the model on full training data
        for epoch in range(num_epochs):
            train_loss, train_r2 = train_one_epoch(model, train_loader_full, criterion, optimizer, device)
        
        # Evaluate on training data
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
        file_path = os.path.join(dir_name, f'{dataType}_{subset}_{target}_cv_results.csv')
        results_df.to_csv(file_path, index=False)
        print(f'Cross-validation results saved to {file_path}')
        
        # Save test set results
        overall_results_df = pd.DataFrame([overall_results])
        overall_file_path = os.path.join(dir_name, f'{dataType}_{subset}_{target}_test_results.csv')
        overall_results_df.to_csv(overall_file_path, index=False)
        print(f'Test results saved to {overall_file_path}')

# Run the cross-validation
simple_cross_validation()
