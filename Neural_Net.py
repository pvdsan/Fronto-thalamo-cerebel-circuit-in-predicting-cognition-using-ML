import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from common_utils import create_kfold_data, normalize_data, train_one_epoch, validate_one_epoch, retrieveDataset, ensure_directory_exists


        

class RegressionNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x
    
    

def main_training_loop(dataType = 'SMRI', subset = 'FTPC', lr=0.00001, batch_size=64, num_epochs=50):
    
    device = torch.device('cuda')
    dataset = retrieveDataset(data = dataType, subset = subset)
    dir_name = f'neural_net_results_{dataType}_{subset}'
    ensure_directory_exists(dir_name)
    
    for target, (X, y) in dataset.items():
        
        print(f'Training for {target}')
        X,y = normalize_data(X  = X, y = y, by = 'std')
        featureLen, folds_data = create_kfold_data(X,y)
        columns = ['fold', 'epoch', 'train_loss', 'val_loss', 'train_r2', 'val_r2']
        all_metrics = pd.DataFrame(columns=columns)

        # Iterate over each fold
        for fold_number, (train_data, val_data) in enumerate(folds_data):
            print(f"Training on fold {fold_number+1}/5")
        
            model = RegressionNet(featureLen).to(device)
            #model = nn.DataParallel(model)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, verbose=True)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers = 8)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers = 8)
            train_losses, val_losses, train_r2_scores, val_r2_scores = [], [], [], []
            
            for epoch in range(num_epochs):
                train_loss, train_r2 = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_r2 = validate_one_epoch(model, val_loader, criterion, device)
                #scheduler.step(val_loss)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_r2_scores.append(train_r2)
                val_r2_scores.append(val_r2)
                
                metrics_data = pd.DataFrame({
                        'fold': [fold_number + 1],
                        'epoch': [epoch + 1],
                        'train_loss': [train_loss],
                        'val_loss': [val_loss],
                        'train_r2': [train_r2],
                        'val_r2': [val_r2]
                })
                    
                all_metrics = pd.concat([all_metrics, metrics_data], ignore_index=True)
                print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train R2: {train_r2}, Val R2: {val_r2}')
            print('--------------------------------------------------------------------------------------------------')
            

            file_path = os.path.join(dir_name, f'{dataType}_{subset}_{target}_training_metrics.csv')
            all_metrics.to_csv(file_path, index=False)

main_training_loop(lr = 1e-4, batch_size=33, num_epochs=200)