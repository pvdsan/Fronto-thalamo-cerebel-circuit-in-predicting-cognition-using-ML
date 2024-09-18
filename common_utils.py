import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import pandas as pd 
from torch.utils.data import DataLoader, TensorDataset
import os

def normalize_data(X, y, by = 'std'):
    
    X = X.to_numpy()
    y = y.to_numpy()
    # Initialize the StandardScaler
    
    if by == 'std':
        scaler = StandardScaler()
    elif by=='minmax':
        scaler = MinMaxScaler()
    
    # Normalize the feature data
    X_scaled = scaler.fit_transform(X)
    
    # Reshape y to a 2D array and normalize
    y = y.reshape(-1, 1)
    Y_scaled = scaler.fit_transform(y).ravel()
    
    # Convert to torch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(Y_scaled, dtype=torch.float32).view(-1, 1)
    
    return X_tensor, y_tensor


def create_kfold_data(X_tensor, y_tensor, n_splits=5):
    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # This will store each fold's train and validation data
    folds_data = []
    
    for train_idx, val_idx in kf.split(X_tensor):
        X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
        X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]
        
        # Create TensorDataset for train and validation
        train_data = TensorDataset(X_train, y_train)
        val_data = TensorDataset(X_val, y_val)
        
        # Append the pair of TensorDatasets to the folds_data list
        folds_data.append((train_data, val_data))
    
    return X_tensor.shape[1], folds_data



def ensure_directory_exists(directory):
    """
    Checks if a directory exists, and if not, creates it.
    
    Args:
        directory (str): The path of the directory to check and create if necessary.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")



def train_one_epoch(model, train_loader, criterion, optimizer, device):
    
    model.train()
    train_loss, train_preds, train_targets = 0, [], []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_preds.append(outputs)
        train_targets.append(targets)
    train_loss /= len(train_loader)
    train_preds = torch.cat(train_preds)
    train_targets = torch.cat(train_targets)
    train_r2 = r2_score(train_targets.cpu().detach().numpy(), train_preds.cpu().detach().numpy())
    return train_loss, train_r2

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss, val_preds, val_targets = 0, [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_preds.append(outputs)
            val_targets.append(targets)
    val_loss /= len(val_loader)
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    val_r2 = r2_score(val_targets.cpu().detach().numpy(), val_preds.cpu().detach().numpy())
    return val_loss, val_r2


def validate_one_epoch2(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            running_loss += loss.item() * X_batch.size(0)
            
            # Collect predictions and true values
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Compute R² score
    r2 = r2_score(y_true, y_pred)
    # Compute Pearson correlation coefficient
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    return epoch_loss, r2, corr


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