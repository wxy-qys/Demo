import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import data, color, transform, exposure
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import warnings
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


class SpikingNeuron(nn.Module):
    def __init__(self, threshold: float = 1.0, decay: float = 0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = None
        self.spike_count = None
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.membrane_potential is None or self.membrane_potential.shape != x.shape:
            self.membrane_potential = torch.zeros_like(x, device=x.device)
            self.spike_count = torch.zeros_like(x, device=x.device)
        
        self.membrane_potential = self.membrane_potential * self.decay + x
        
        spikes_hard = (self.membrane_potential >= self.threshold).float()
        
        surrogate = torch.sigmoid(5 * (self.membrane_potential - self.threshold))
        spikes = surrogate + (spikes_hard - surrogate).detach()
        
        self.membrane_potential = self.membrane_potential * (1 - spikes_hard)
        self.spike_count += spikes_hard
        
        return spikes, self.membrane_potential, self.spike_count
    
    def reset(self):
        if self.membrane_potential is not None:
            self.membrane_potential.zero_()
            if self.spike_count is not None:
                self.spike_count.zero_()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)
        
        avg_att = self.fc(avg_out).view(B, C, 1, 1)
        max_att = self.fc(max_out).view(B, C, 1, 1)
        
        att = avg_att + max_att
        return x * att


class SpikeDecoderWithHybridAttention(nn.Module):
    def __init__(self, hidden_dim: int = 64, out_channels: int = 1):
        super().__init__()
        
        self.res1 = ResidualBlock(hidden_dim * 8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True)
        )
        self.fusion1 = nn.Conv2d(hidden_dim * 8, hidden_dim * 4, 1)
        self.attn1 = ChannelAttention(hidden_dim * 4, reduction=4)
        self.res2 = ResidualBlock(hidden_dim * 4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        self.fusion2 = nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 1)
        self.attn2 = ChannelAttention(hidden_dim * 2, reduction=4)
        self.res3 = ResidualBlock(hidden_dim * 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.fusion3 = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)
        self.attn3 = ChannelAttention(hidden_dim, reduction=4)
        self.res4 = ResidualBlock(hidden_dim)
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        x = self.res1(x)
        x = self.conv1(x)
        
        x = self.up1(x)
        skip = skip_features[-1]
        x = torch.cat([x, skip], dim=1)
        x = self.fusion1(x)
        x = self.attn1(x)
        x = self.res2(x)
        x = self.conv2(x)
        
        x = self.up2(x)
        skip = skip_features[-2]
        x = torch.cat([x, skip], dim=1)
        x = self.fusion2(x)
        x = self.attn2(x)
        x = self.res3(x)
        x = self.conv3(x)
        
        x = self.up3(x)
        skip = skip_features[-3]
        x = torch.cat([x, skip], dim=1)
        x = self.fusion3(x)
        x = self.attn3(x)
        x = self.res4(x)
        x = self.conv4(x)
        
        x = self.final_conv(x)
        
        return x


class AttentionMechanism(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        residual = x
        x = self.norm(x)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x + residual
        
        return x, attn


class LinearAttentionMechanism(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.elu = nn.ELU()
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = x.shape
        residual = x
        x = self.norm(x)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q = self.elu(q)
        k = self.elu(k)
        
        kv = k.transpose(-2, -1) @ v
        x = q @ kv
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x + residual
        
        return x, None


class SpikeEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True)
        )
        self.res_block1 = ResidualBlock(hidden_dim * 4)
        self.res_block2 = ResidualBlock(hidden_dim * 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.spiking = SpikingNeuron()
        
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 1),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 1),
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_features = []
        
        x = self.conv1(x)
        skip_features.append(x)
        x, mem, spikes = self.spiking(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        skip_features.append(x)
        x, mem, spikes = self.spiking(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.res_block1(x)
        skip_features.append(x)
        x, mem, spikes = self.spiking(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.res_block2(x)
        x, mem, spikes = self.spiking(x)
        
        return x, skip_features


class SpikeDecoderWithCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int = 64, out_channels: int = 1):
        super().__init__()
        
        self.res1 = ResidualBlock(hidden_dim * 8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True)
        )
        self.cross_attn1 = CrossAttention(hidden_dim * 4, hidden_dim * 4, num_heads=4)
        self.res2 = ResidualBlock(hidden_dim * 4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        self.cross_attn2 = CrossAttention(hidden_dim * 2, hidden_dim * 2, num_heads=4)
        self.res3 = ResidualBlock(hidden_dim * 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.cross_attn3 = CrossAttention(hidden_dim, hidden_dim, num_heads=4)
        self.res4 = ResidualBlock(hidden_dim)
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        B = x.shape[0]
        
        x = self.res1(x)
        x = self.conv1(x)
        
        x = self.up1(x)
        H, W = x.shape[2], x.shape[3]
        skip = skip_features[-1]
        skip_flat = skip.flatten(2).transpose(1, 2)
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.cross_attn1(x_flat, skip_flat)
        x = x_flat.transpose(1, 2).reshape(B, -1, H, W)
        x = self.res2(x)
        x = self.conv2(x)
        
        x = self.up2(x)
        H, W = x.shape[2], x.shape[3]
        skip = skip_features[-2]
        skip_flat = skip.flatten(2).transpose(1, 2)
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.cross_attn2(x_flat, skip_flat)
        x = x_flat.transpose(1, 2).reshape(B, -1, H, W)
        x = self.res3(x)
        x = self.conv3(x)
        
        x = self.up3(x)
        H, W = x.shape[2], x.shape[3]
        skip = skip_features[-3]
        skip_flat = skip.flatten(2).transpose(1, 2)
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.cross_attn3(x_flat, skip_flat)
        x = x_flat.transpose(1, 2).reshape(B, -1, H, W)
        x = self.res4(x)
        x = self.conv4(x)
        
        x = self.final_conv(x)
        
        return x


class SpikeImageReconstructionModel(nn.Module):
    def __init__(self, img_size: int = 128, hidden_dim: int = 64, num_heads: int = 4, use_linear_attention: bool = False):
        super().__init__()
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.use_linear_attention = use_linear_attention
        
        self.encoder = SpikeEncoder(in_channels=1, hidden_dim=hidden_dim)
        
        if use_linear_attention:
            self.attention = LinearAttentionMechanism(hidden_dim * 8, num_heads)
            print("使用Linear注意力机制")
        else:
            self.attention = AttentionMechanism(hidden_dim * 8, num_heads)
            print("使用标准注意力机制")
        
        self.decoder = SpikeDecoderWithHybridAttention(hidden_dim=hidden_dim, out_channels=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = x.size(0)
        
        encoded, skip_features = self.encoder(x)
        
        B, C, H, W = encoded.shape
        encoded_flat = encoded.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        attended, attn_matrix = self.attention(encoded_flat)
        
        attended = attended.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        reconstructed = self.decoder(attended, skip_features)
        
        return reconstructed, attn_matrix
    
    def reset_neurons(self):
        for module in self.modules():
            if isinstance(module, SpikingNeuron):
                module.reset()


def create_high_res_dataset(num_samples: int = 50, img_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    images = []
    spike_trains = []
    
    base_images = [
        data.camera(),
        data.coins(),
        data.astronaut(),
        data.chelsea(),
        data.horse(),
        data.coffee(),
        data.page(),
    ]
    
    for i in range(num_samples):
        base_img = base_images[i % len(base_images)]
        
        if len(base_img.shape) == 3:
            img = color.rgb2gray(base_img)
        else:
            img = base_img.astype(np.float64)
        
        if img.dtype == bool:
            img = img.astype(np.float64)
        
        img = transform.resize(img, (img_size, img_size), anti_aliasing=True)
        
        contrast = np.random.uniform(0.8, 1.2)
        brightness = np.random.uniform(-0.1, 0.1)
        img = exposure.adjust_gamma(img, contrast)
        img = np.clip(img + brightness, 0, 1)
        
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        images.append(img)
        
        threshold = np.random.uniform(0.3, 0.6)
        spikes = (img > threshold).astype(np.float32)
        spike_trains.append(spikes)
    
    images = torch.tensor(np.array(images), dtype=torch.float32).unsqueeze(1)
    spike_trains = torch.tensor(np.array(spike_trains), dtype=torch.float32).unsqueeze(1)
    
    return spike_trains, images


class SpikeDataset(Dataset):
    def __init__(self, spike_trains: torch.Tensor, images: torch.Tensor):
        self.spike_trains = spike_trains
        self.images = images
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.spike_trains[idx], self.images[idx]


def perceptual_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean(dim=[2, 3], keepdim=True)
    target_mean = target.mean(dim=[2, 3], keepdim=True)
    mean_loss = ((pred_mean - target_mean) ** 2).mean()
    
    pred_std = pred.std(dim=[2, 3], keepdim=True)
    target_std = target.std(dim=[2, 3], keepdim=True)
    std_loss = ((pred_std - target_std) ** 2).mean()
    
    return mean_loss + std_loss


def second_order_hotv(pred: torch.Tensor) -> torch.Tensor:
    ddy2 = pred[:, :, 2:, :] - 2 * pred[:, :, 1:-1, :] + pred[:, :, :-2, :]
    ddx2 = pred[:, :, :, 2:] - 2 * pred[:, :, :, 1:-1] + pred[:, :, :, :-2]
    dxdy = pred[:, :, 1:, 1:] - pred[:, :, 1:, :-1] - pred[:, :, :-1, 1:] + pred[:, :, :-1, :-1]
    
    hotv = torch.mean(torch.abs(ddy2)) + torch.mean(torch.abs(ddx2)) + 0.5 * torch.mean(torch.abs(dxdy))
    return hotv


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = nn.functional.mse_loss(pred, target)
    perceptual = perceptual_loss(pred, target)
    hotv = second_order_hotv(pred)
    
    loss = 0.6 * mse + 0.25 * perceptual + 0.15 * hotv
    return loss


def visualize_results(original: torch.Tensor, reconstructed: torch.Tensor, num_samples: int = 4):
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 8))
    
    for i in range(num_samples):
        axes[0, i].imshow(original[i, 0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        rec_img = reconstructed[i, 0].detach().cpu().numpy()
        rec_img = np.clip(rec_img, 0, 1)
        axes[1, i].imshow(rec_img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_results_cross_attn.png', dpi=200)
    print("结果已保存到 reconstruction_results_cross_attn.png")
    plt.close()
    
    mse_loss = nn.functional.mse_loss(reconstructed, original).item()
    psnr = 10 * np.log10(1.0 / mse_loss) if mse_loss > 0 else float('inf')
    print(f"MSE: {mse_loss:.6f}, PSNR: {psnr:.2f} dB")


def train_model(model: nn.Module, dataloader: DataLoader, num_epochs: int = 50, lr: float = 1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    model.train()
    best_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0
        for spikes, targets in dataloader:
            spikes = spikes.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            model.reset_neurons()
            
            reconstructed, _ = model(spikes)
            loss = combined_loss(reconstructed, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model_cross_attn.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Best Loss: {best_loss:.6f}")
    
    print(f"\n训练完成，最佳损失: {best_loss:.6f}")
    model.load_state_dict(torch.load('best_model_cross_attn.pth'))
    
    return model


def main():
    print("初始化交叉注意力脉冲图像重建模型...")
    print(f"PyTorch版本: {torch.__version__}")
    
    img_size = 128
    hidden_dim = 64
    batch_size = 16
    num_epochs = 100
    
    print("\n生成数据集...")
    spike_trains, images = create_high_res_dataset(num_samples=200, img_size=img_size)
    dataset = SpikeDataset(spike_trains, images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"数据集大小: {len(dataset)}, 图像分辨率: {img_size}x{img_size}, Batch Size: {batch_size}")
    
    model = SpikeImageReconstructionModel(
        img_size=img_size,
        hidden_dim=hidden_dim,
        num_heads=4,
        use_linear_attention=False
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量: {total_params:,}")
    
    print("\n开始训练模型（交叉注意力机制）...")
    model = train_model(model, dataloader, num_epochs=num_epochs, lr=5e-4)
    
    model.eval()
    model.reset_neurons()
    
    with torch.no_grad():
        sample_spikes, sample_images = next(iter(dataloader))
        sample_spikes = sample_spikes.to(DEVICE)
        sample_images = sample_images.to(DEVICE)
        reconstructed, _ = model(sample_spikes)
        
        num_display = min(4, sample_images.size(0))
        visualize_results(sample_images, reconstructed, num_samples=num_display)
    
    print("\n训练完成！")
    
    torch.save(model.state_dict(), 'spike_reconstruction_model_cross_attn.pth')
    print("模型已保存到 spike_reconstruction_model_cross_attn.pth")


if __name__ == "__main__":
    main()
