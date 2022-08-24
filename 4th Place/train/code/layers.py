import math
import torch
import torch.nn as nn
import torch.fft

from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parameter import Parameter

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels,
            torch.div(x.size(-1), 2, rounding_mode='floor') + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class SpectralBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes,
                 kernel_size=1, stride=1, bias=False, activator=nn.ReLU):
        super(SpectralBlock1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # Fourier domain
        self.spectr = SpectralConv1d(self.in_channels, self.out_channels,
                                            self.modes)
        # Feature domain
        self.linear = nn.Conv1d(self.in_channels, self.out_channels,
                                kernel_size=kernel_size, padding=kernel_size//2,
                                stride=stride, bias=bias)
        # Normalize
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.activator = activator()

    def forward(self, x):
        # Features domain forward
        x2 = self.linear(x)
        # Fourier domain forward
        x1 = self.spectr(x)
        # Add time and feature
        x = self.bn(x1 + x2)
        x = self.activator(x)

        return x

class SnowNet(nn.Module):
    def __init__(self, features=6, h_dim=32,
                 width=48, timelag=21, out_dim=1, embedding_dim = 3):

        super(SnowNet, self).__init__()
        self.features = features
        self.timelag = timelag
        self.modes, self.width = width // 2, width
        self.h_dim, self.in_dim = h_dim, timelag
        self.k = 2
        self.embs = nn.Embedding(num_embeddings = 20,
                                 embedding_dim = embedding_dim)
        self.embs.weight.data[0] *= 0

        if timelag == width:
            self.fc0 = nn.Identity()
        else:
            self.fc0 = nn.Linear(self.timelag, self.width)

        self.step0 = nn.Sequential(
            nn.Conv1d(self.features, self.h_dim, 1, bias = False),
            nn.BatchNorm1d(self.h_dim), nn.PReLU(),
            nn.Conv1d(self.h_dim, self.h_dim, 1),
        )

        # Conv layers:
        self.conv0dem = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, bias=False),
            nn.BatchNorm2d(6), nn.PReLU(),
            nn.Conv2d(in_channels = 6, out_channels = 9, kernel_size = 3, bias = False),
            nn.BatchNorm2d(9), nn.PReLU(),
            nn.Conv2d(in_channels = 9, out_channels = 6, kernel_size = 3, bias = False),
            nn.Flatten(),
            nn.Linear(6*4*4, self.width),
        )

        self.conv1soil = nn.Sequential(
            nn.Conv2d(embedding_dim, 6, kernel_size=3, bias=False),
            nn.BatchNorm2d(6), nn.PReLU(),
            nn.Conv2d(in_channels = 6, out_channels = 9, kernel_size = 3, bias = False),
            nn.BatchNorm2d(9), nn.PReLU(),
            nn.Conv2d(in_channels = 9, out_channels = 6, kernel_size = 3, bias = False),
            nn.Flatten(),
            nn.Linear(6*4*4, self.width),
        )

        self.layers = nn.Sequential(
            SpectralBlock1d(self.h_dim + 2, self.h_dim, int(self.modes*1/2),
                            activator=nn.PReLU),
            SpectralBlock1d(self.h_dim, self.h_dim, int(self.modes*1/2),
                            activator=nn.PReLU),
            SpectralBlock1d(self.h_dim, self.h_dim, int(self.modes*1/2),
                            activator=nn.PReLU),
            SpectralBlock1d(self.h_dim, self.h_dim, int(self.modes*1/4),
                            activator=nn.PReLU),
        )

        self.step_t = nn.Conv1d(self.h_dim, 1, 1)
        self.fc1 = nn.Linear(width, out_dim)

    def forward(self, x, dem, soil):

        x = self.fc0(x)
        x = self.step0(x)

        soil = self.embs(soil.long()).permute(0,3,1,2)
        d0 = self.conv0dem(dem)
        d1 = self.conv1soil(soil)
        x = torch.cat([
            x, d0.view(-1, 1, self.width), d1.view(-1, 1, self.width)], 1)

        x = self.layers(x)

        x = self.step_t(x)
        x = self.fc1(x)

        return x.squeeze(-1)

class ModelAggregator(nn.Module):
    def __init__(self, models):
        super(ModelAggregator, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args):
        x_a = torch.stack([
            m(*args) for m in self.models], dim=-1)
        return x_a.clamp(0).mean(-1)

def get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, min_coef=0.0, last_epoch=-1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            min_coef, float(num_training_steps - current_step) /  float(
                                max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int,
    num_cycles: float = 0.5, min_coef: float = 0.0, last_epoch: int = -1
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
                                                max(1, num_training_steps - num_warmup_steps))
        return max(min_coef, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
