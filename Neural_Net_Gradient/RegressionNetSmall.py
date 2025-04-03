import torch.nn as nn 
import torch 
    

    
class RegressionNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.dropout1 = nn.Dropout(0.2)
        self.output = nn.Linear(200, 1)        

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.output(x)
        return x