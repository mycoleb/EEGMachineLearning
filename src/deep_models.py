import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """
    EEGNet implementation for motor imagery classification.
    Based on the paper: EEGNet: A Compact Convolutional Network for EEG-based BCIs
    """
    def __init__(self, n_channels=64, n_classes=3, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        
        # Layer 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(8)
        
        # Layer 2: Spatial Convolution
        self.conv2 = nn.Conv2d(8, 16, (n_channels, 1), padding=0)
        self.batchnorm2 = nn.BatchNorm2d(16)
        
        # Layer 3: Separable Convolution
        self.conv3 = nn.Conv2d(16, 32, (1, 16), padding='same', groups=16)
        self.batchnorm3 = nn.BatchNorm2d(32)
        
        # Average Pooling
        self.avgpool = nn.AvgPool2d((1, 4))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense Layer
        self.fc = nn.Linear(32 * 16, n_classes)  # Adjust size based on input
    
    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        
        # Flatten and Dense
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return F.softmax(x, dim=1)

class DeepConvNet(nn.Module):
    """
    Deep Convolutional Network for EEG classification.
    """
    def __init__(self, n_channels=64, n_classes=3, dropout_rate=0.5):
        super(DeepConvNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5)),
            nn.Conv2d(25, 25, (n_channels, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout_rate)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 5)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout_rate)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 5)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout_rate)
        )
        
        # Dense Layer
        self.fc = nn.Linear(100 * 8, n_classes)  # Adjust size based on input
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return F.softmax(x, dim=1)

def train_model(model, train_loader, valid_loader, criterion, optimizer, n_epochs=50, device='cuda'):
    """
    Train the deep learning model.
    """
    model = model.to(device)
    best_valid_loss = float('inf')
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        valid_loss /= len(valid_loader)
        accuracy = correct / len(valid_loader.dataset)
        
        print(f'Epoch: {epoch}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {valid_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.4f}\n')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    return model