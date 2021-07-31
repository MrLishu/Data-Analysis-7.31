import torch.nn as nn


class TINet(nn.Module):
    def __init__(self, C_in, class_num):
        super(TINet, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(C_in, 16, 64, stride=8, padding=27, bias=False),
            nn.Dropout(0.3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 100),
            nn.Linear(100, class_num),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        features = self.extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out


class SiameseNet(nn.Module):
    def __init__(self, C_in, class_num):
        super(SiameseNet, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=64, stride=16, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Dense layer
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=100),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=100, out_features=class_num),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.net(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = self.classifier(output)
        return output
