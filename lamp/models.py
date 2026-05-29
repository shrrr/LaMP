import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2sse = lambda x, y : torch.sum((x - y) ** 2)
img2psse = lambda x, y, N : torch.sum((x.reshape(-1,N) - y.reshape(-1,N)).abs() ** 2, dim=0)
img2pss = lambda x, N : torch.sum((x.reshape(-1,N).abs()) ** 2, dim=0)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 2
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 2,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeJF_bak(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, output_ch=1, skips=[4], tanh=None, scale=1e-5, epsilon=False):
        """
        """
        super(NeJF_bak, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.tanh = tanh
        self.epsilon = epsilon
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

        self.output_linear = nn.Linear(W, output_ch)
        self.scale = scale

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, 0], dim=-1)
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = self.output_linear(h)
        if self.tanh is not None:
            if self.epsilon:
                outputs = 0.5 * (torch.tanh(outputs) + 1) + 1 
            else:
                outputs = outputs * self.scale
        return outputs

class NeJF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, output_ch=1, skips=[4], tanh=None, scale=1e-5, epsilon=False):
        """
        """
        super(NeJF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.tanh = tanh
        self.epsilon = epsilon
        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
        #                                 range(D - 1)])
        self.pts_linears = nn.ModuleList()
        in_dim = input_ch
        for i in range(D):
            if i-1 in skips:
                in_dim += input_ch  # 跳跃连接维度拼接
            self.pts_linears.append(
                nn.Sequential(
                    nn.Linear(in_dim, W),
                    nn.BatchNorm1d(W),  # 归一化前置
                    nn.GELU(),  # 改进2：GELU代替ReLU
                    nn.Dropout(0.1) if i % 2 == 0 else nn.Identity()  # 改进3：交替Dropout
                )
            )
            in_dim = W

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

        # self.output_linear = nn.Linear(W, output_ch)
        self.output_linear = nn.Sequential(
            nn.Linear(W, W//2),
            nn.BatchNorm1d(W//2),
            nn.SiLU(),
            nn.Linear(W//2, output_ch)
        )
        self.scale = scale

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, 0], dim=-1)
        h = x
        for i, l in enumerate(self.pts_linears):
            if i == 0:
                h = self.pts_linears[i](h)
                h_shortcut = h
            elif i % 3 == 0:
                h = (self.pts_linears[i](h)+h_shortcut)/2
                h_shortcut = h
            else:
                h = self.pts_linears[i](h)
            # h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = self.output_linear(h)
        if self.tanh is not None:
            if self.epsilon:
                outputs = 0.5 * (torch.tanh(outputs) + 1) + 1 
            else:
                outputs = outputs * self.scale
        return outputs

class NeEtotF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, output_ch=1, skips=[4], tanh=None, scale=1e-5):
        """
        """
        super(NeEtotF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.tanh = tanh
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

        self.output_linear = nn.Linear(W, output_ch)
        self.scale = scale

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.tanh(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        outputs = self.output_linear(h)
        if self.tanh is not None:
            # outputs = 1.6 * (torch.tanh(outputs) + 1) # for real data
            # outputs = 0.0025 * (torch.tanh(outputs))  # 1.6
            # outputs = outputs * self.scale
            outputs = 1 * (torch.tanh(outputs))
        return outputs

class NeJF_3D(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, output_ch=1, skips=[4], tanh=None):
        """
        """
        super(NeJF_3D, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.tanh = tanh
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, 0], dim=-1)
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = self.output_linear(h)
        if self.tanh is not None:
            # outputs = 1.6 * (torch.tanh(outputs) + 1) # for real data
            # outputs = 0.5 * (torch.tanh(outputs) + 1) + 1  # 1.6 # for circle
            # outputs = torch.tanh(outputs)*1.5  # [1, 2] # for circle
            # outputs = torch.sigmoid(outputs)  # [1, 3] for mnist
            outputs = outputs * 1e-5
        return outputs

class AdaptiveAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.1))
    def forward(self, x):
        return self.alpha * torch.tanh(self.beta * x)
    
class FrequencyBranch(nn.Module):
    def __init__(self, layer_num, hidden_dim, input_dim=2, output_dim=2, skips=[4], learnable_tanh=False, scale=1e-4):
        super().__init__()
        self.skips = skips
        self.scale = scale
        self.learnable_tanh = learnable_tanh
        self.input_dim = input_dim
        
        if learnable_tanh:
            self.activation = AdaptiveAct()
        else:
            self.activation = nn.ReLU()
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i not in self.skips else nn.Linear(hidden_dim + input_dim, hidden_dim) for i in
                                        range(layer_num - 1)])
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_dim, 0], dim=-1)
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.activation(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        outputs = self.output_linear(h)
        outputs = outputs * self.scale
        return outputs

class MultiFrequencyNet(nn.Module):
    """主网络架构：共享特征+注意力+多频率分支"""
    def __init__(self, layer_num, num_freqs, input_dim=3, output_dim=2, shared_dim=64, skips=[4], learnable_tanh=False, inc_num=16, baselayer_num=4):
        super().__init__()
        self.inc_num = inc_num
        self.shared_encoder = FrequencyBranch(baselayer_num, shared_dim, input_dim, output_dim=shared_dim, skips=[2], scale=1)
        # 多头注意力层（空间坐标特征增强）
        self.coord_attention = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4)
        self.inc_attention = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=2)
        # 频率分支（每个分支对应一个频率）
        self.branches = nn.ModuleList([
            FrequencyBranch(layer_num, shared_dim, shared_dim, output_dim=2, skips=[4,8,12,16,20]) for _ in range(num_freqs)
        ])
    
    def forward(self, x):
        # 输入x: (batch_size, 3) → (x,y,z坐标)
        shared_feat = self.shared_encoder(x)
        # 注意力增强特征（增强空间关联性）
        shared_feat_flatten = shared_feat.reshape(-1, self.inc_num, shared_feat.shape[-1])
        coord_attn_feat, _ = self.coord_attention(shared_feat_flatten, shared_feat_flatten, shared_feat_flatten)
        inc_attn_feat, _ = self.inc_attention(shared_feat_flatten.transpose(0, 1), shared_feat_flatten.transpose(0, 1), shared_feat_flatten.transpose(0, 1))
        # 残差拼接
        attn_feat = shared_feat + coord_attn_feat.reshape(-1, shared_feat.shape[-1]) + inc_attn_feat.transpose(0, 1).reshape(-1, shared_feat.shape[-1])
        # attn_feat = shared_feat
        # 多频率分支预测
        outputs = []
        for i, branch in enumerate(self.branches):
            y_pred = branch(attn_feat)
            outputs.append(y_pred)
        return torch.stack(outputs, dim=1).transpose(0, 1)

class EpsilonNet(nn.Module):
    def __init__(self, init_epsilon, use_constraint=False):
        super().__init__()
        self.epsilon = nn.Parameter(init_epsilon, requires_grad=True)
        self.use_constraint = use_constraint
    
    def forward(self):
        if self.use_constraint:
            self.epsilon = F.tanh(self.epsilon)
            self.epsilon = 0.5 * (self.epsilon + 1) + 1
        return self.epsilon

def epsilon_control(x, constraint=None):
    if constraint == '12':
        x = F.tanh(x)
        x = 0.5 * (x + 1) + 1
    elif constraint == 'positive':
        x = F.softplus(x-1) + 1
    elif constraint == '14':
        x = 1 + 3*F.sigmoid(x)
    else:
        pass
    return x