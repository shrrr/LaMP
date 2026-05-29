import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae_finetune import *

class ConvVAE(nn.Module):
    def __init__(self, img_channels=1, latent_dim=128, hidden_dims=[16, 32, 64, 128], **unused):
        super(ConvVAE, self).__init__()
        
        # Encoder
        modules = []
        in_channels = img_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(2, 2)
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        
        # 计算展平后的特征维度
        self.flatten_dim = hidden_dims[-1] * 4 * 4
        
        # Latent space
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        modules = []
        hidden_dims.reverse()
        for i in range(0, len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU(),
                    nn.Conv2d(hidden_dims[i+1], hidden_dims[i+1], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(hidden_dims[-1], img_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(img_channels),
                nn.LeakyReLU(),
                nn.Conv2d(img_channels, img_channels, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.tanh(self.decoder_input(z))
        x = x.view(-1, 128, 4, 4)
        x = self.decoder(x) + 1
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class DepthwiseSeparableConv(nn.Module):
    """ 深度可分离卷积 (来自官方models/ar_ops.py) """
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel, 
                                 padding=kernel//2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation注意力 (来自官方models/ar_ops.py) """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    """ 带SE注意力的残差块 (简化自官方models/ar_ops.py) """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            DepthwiseSeparableConv(channels, channels),
            nn.BatchNorm2d(channels),
            SEBlock(channels)
        )
        
    def forward(self, x):
        return x + self.conv(x)

class NVAE(nn.Module):
    """ 简化版NVAE (4层结构) """
    def __init__(self, img_channels=1, latent_dim=128, hidden_dims=[16, 32, 64, 128], z_dim=8, **unused):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 16, 3, stride=2, padding=1),  # 64->32
            ResidualBlock(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32->16
            ResidualBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 16->8
            ResidualBlock(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), # 16->8
            ResidualBlock(64),
            nn.Conv2d(64, 2*z_dim, 1)  # 潜在变量参数
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, 64, 1),
            ResidualBlock(64),
            nn.Upsample(scale_factor=2),  # 8->16
            DepthwiseSeparableConv(64, 64),
            ResidualBlock(64),
            nn.Upsample(scale_factor=2),  # 8->16
            DepthwiseSeparableConv(64, 32),
            ResidualBlock(32),
            nn.Upsample(scale_factor=2),  # 16->32
            DepthwiseSeparableConv(32, 16),
            ResidualBlock(16),
            nn.Upsample(scale_factor=2),  # 32->64
            nn.Conv2d(16, img_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, x):
        params = self.encoder(x)
        mu, logvar = params.chunk(2, dim=1)
        return mu, logvar
    
    def decode(self, z):
        z = z.view(-1, 8, 4, 4)
        x = self.decoder(z)
        return x
    
    def forward(self, x):
        # 编码过程
        mu, logvar = self.encode(x)
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        # 解码过程
        recon = self.decode(z)
        return recon, mu, logvar

class InversionDecoder(nn.Module):
    def __init__(self, vae_model, ckpt_path, img_channels=1, 
                 latent_dim=128, hidden_dims=[16, 32, 64, 128], 
                 vae_config_path=None, max_params=2, params_epsilon=None):
        super(InversionDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.vae_model = vae_model
        self.max_params = max_params
        vae2model = {
            'vanilla_vae': ConvVAE,
            'optim_vae': NVAE,
            'sd_vae_ft': SDVAEFT,
        }
        self.conv_vae = vae2model[vae_model](img_channels=img_channels, latent_dim=latent_dim, hidden_dims=hidden_dims, vae_config_path=vae_config_path, max_params=max_params)
        self.conv_vae.load_state_dict(torch.load(ckpt_path))
        
        if params_epsilon is not None:
            self.add_params = True
            self.params_epsilon = params_epsilon
        else:
            self.add_params = False
            self.params_epsilon = None
        
        # # 冻结所有参数
        # for param in self.conv_vae.parameters():
        #     param.requires_grad = False
        # self.conv_vae.eval()
    
    def forward(self, z):
        # with torch.no_grad():  # 确保不计算梯度
        if self.vae_model in ['vanilla_vae', 'optim_vae']:
            z = z.view(-1, self.latent_dim)
        
        if self.add_params:
            return (self.conv_vae.decode(z).squeeze()+ self.params_epsilon)/2
        else:
            return self.conv_vae.decode(z).squeeze()
    
    def get_latent(self, x):
        # with torch.no_grad():
        mu, logvar = self.conv_vae.encode(x)
        return mu
        

if __name__ == '__main__':
    test_input = torch.randn(1, 1, 64, 64)
    model = NVAE()
    recon, mu, logvar = model(test_input)
    print(f"输入尺寸: {test_input.shape}")
    print(f"重建输出尺寸: {recon.shape}")
    print(f"潜在变量均值尺寸: {mu.shape}")