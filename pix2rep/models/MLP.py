import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class MLP(nn.Module):
    def __init__(self, channels_in, channels_out, inner_dim_1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, inner_dim_1, 1),
            nn.BatchNorm2d(inner_dim_1),
            nn.ReLU(),
            nn.Conv2d(inner_dim_1, channels_out, 1),
        )

    def forward(self, x):
        return self.net(x)



class ConvMLP_3_layers(nn.Module):
    def __init__(self, channels_in, channels_out, inner_dim_1, inner_dim_2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, inner_dim_1, 1),
            nn.BatchNorm2d(inner_dim_1),
            nn.ReLU(),
            nn.Conv2d(inner_dim_1, inner_dim_2, 1),
            nn.BatchNorm2d(inner_dim_2),
            nn.ReLU(),
            nn.Conv2d(inner_dim_2, channels_out, 1),
        )

    def forward(self, x):
        return self.net(x)


class ConvMLP_4_layers(nn.Module):
    def __init__(self, channels_in, channels_out, inner_dim_1, inner_dim_2, inner_dim_3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, inner_dim_1, 1),
            nn.BatchNorm2d(inner_dim_1),
            nn.ReLU(),
            nn.Conv2d(inner_dim_1, inner_dim_2, 1),
            nn.BatchNorm2d(inner_dim_2),
            nn.ReLU(),
            nn.Conv2d(inner_dim_2, inner_dim_3, 1),
            nn.BatchNorm2d(inner_dim_3),
            nn.ReLU(),
            nn.Conv2d(inner_dim_3, channels_out, 1),
        )

    def forward(self, x):
        return self.net(x)


class ConvMLP_5_layers(nn.Module):
    def __init__(self, channels_in, channels_out, inner_dim_1, inner_dim_2, inner_dim_3, inner_dim_4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, inner_dim_1, 1),
            nn.BatchNorm2d(inner_dim_1),
            nn.ReLU(),
            nn.Conv2d(inner_dim_1, inner_dim_2, 1),
            nn.BatchNorm2d(inner_dim_2),
            nn.ReLU(),
            nn.Conv2d(inner_dim_2, inner_dim_3, 1),
            nn.BatchNorm2d(inner_dim_3),
            nn.ReLU(),
            nn.Conv2d(inner_dim_3, inner_dim_4, 1),
            nn.BatchNorm2d(inner_dim_4),
            nn.ReLU(),
            nn.Conv2d(inner_dim_4, channels_out, 1),
        )

    def forward(self, x):
        return self.net(x)
