class BrainDigiCNN(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(BrainDigiCNN, self).__init__()
        
        # 4 Conv1D layers sesuai paper
        self.conv1 = nn.Conv1d(1, 256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(256, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        
        self.conv4 = nn.Conv1d(64, 32, kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.MaxPool1d(2)
        
        # 2 FC layers sesuai paper
        conv_output_size = 32 * (input_size // 16)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv layers: Conv1D → BN → ReLU → MaxPool
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten + FC layers
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # SoftMax applied in loss
        return x