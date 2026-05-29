import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import pandas as pd
from tabulate import tabulate

class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # batch_size = x.size()[0]
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h =  (x.size()[0]-1) * x.size()[1]
        count_w = x.size()[0] * (x.size()[1] - 1)
        h_tv = torch.pow((x[1:,:]-x[:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,1:]-x[:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)
    
class L_TV_L1(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV_L1,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # batch_size = x.size()[0]
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h =  (x.size()[0]-1) * x.size()[1]
        count_w = x.size()[0] * (x.size()[1] - 1)
        h_tv = torch.abs((x[1:,:]-x[:h_x-1,:])).sum()
        w_tv = torch.abs((x[:,1:]-x[:,:w_x-1])).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)

class L_TV_L2(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV_L2, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        # x shape: [H, W]
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        count_total = count_h + count_w
        diff_h = x[1:, :] - x[:-1, :]
        diff_w = x[:, 1:] - x[:, :-1]
        diff_h_pad = F.pad(diff_h, (0, 0, 0, 1))
        diff_w_pad = F.pad(diff_w, (0, 1, 0, 0))
        tv_val = torch.sqrt(torch.pow(diff_h_pad, 2) + torch.pow(diff_w_pad, 2) + 1e-12).sum()
        return self.TVLoss_weight * 2 * (tv_val / count_total)

def get_latest_testset_paths(folder_list):
    latest_paths = []
    for dir_path in folder_list:
        if not os.path.exists(dir_path):
            print(f"Warning: 目录不存在 -> {dir_path}")
            continue
            
        search_pattern = os.path.join(dir_path, 'testset*.npy')
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"Warning: 在目录中未找到 testset 文件 -> {dir_path}")
            continue
            
        latest_file = max(files, key=os.path.getmtime)
        latest_paths.append(latest_file)

    return latest_paths

# class MultiDirectionalTV(nn.Module):
#     def __init__(self, n_directions=8, weight=1.0):
#         super().__init__()
#         self.weight = weight
#         self.directions = self._get_directions(n_directions)

#     def _get_directions(self, n):
#         """生成9个方向的偏移量 (dy, dx)"""
#         if n == 8:  # 8方向
#             return [ (0,1), (1,0), (1,1), (1,-1), 
#                     (0,-1), (-1,0), (-1,1), (-1,-1)]
#         elif n == 9: # 9方向(包含零边距)
#             return [ (0,1), (1,0), (1,1), (1,-1),
#                      (0,-1), (-1,0), (-1,1), (-1,-1),
#                      (0,0) ] # 中心（需要特殊处理）
#         else:
#             raise ValueError("支持8或9个方向")

#     def forward(self, x):
#         """
#         输入: x 张量 [B,C,H,W] 或 [H,W]
#         输出: 正则化损失值
#         """
#         if x.dim() == 2:
#             x = x.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
#         elif x.dim() == 3:
#             raise ValueError("不支持3D张量，需为[B,C,H,W]或[H,W]")
#         B, C, H, W = x.shape
#         total_tv = 0.0
#         for (dy, dx) in self.directions:
#             # 跳过中心方向的无效梯度计算
#             if dy == 0 and dx == 0:
#                 continue
#             # 计算有效区域边界
#             h_start = max(-dy, 0)
#             h_end = H - max(dy, 0)
#             w_start = max(-dx, 0)
#             w_end = W - max(dx, 0)
#             # 提取对比区域
#             source = x[:, :, h_start:h_end, w_start:w_end]
#             target = x[:, :, 
#                      h_start+dy:h_end+dy, 
#                      w_start+dx:w_end+dx]
#             # 计算绝对差分并归一化
#             valid_pixels = B * C * (h_end-h_start) * (w_end-w_start)
#             if valid_pixels == 0:
#                 continue
#             total_tv += torch.abs(source - target).sum() / valid_pixels
#         return self.weight * total_tv


class MultiDirectionalTV(nn.Module):
    def __init__(self, main_directions=[(0,1), (1,0)], weight=1.0, gamma=0.7):
        super().__init__()
        self.main_directions = main_directions  # 主边缘方向
        self.other_directions = [(1,1), (1,-1), (0,-1), (-1,0), (-1,1), (-1,-1)]
        self.weight = weight
        self.gamma = gamma  # 主方向权重衰减系数

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        B, C, H, W = x.shape
        total_tv = 0.0
        
        # 主方向计算 (低惩罚)
        for dy, dx in self.main_directions:
            h_start = max(-dy, 0)
            h_end = H - max(dy, 0)
            w_start = max(-dx, 0)
            w_end = W - max(dx, 0)
            # 提取对比区域
            source = x[:, :, h_start:h_end, w_start:w_end]
            target = x[:, :, h_start+dy:h_end+dy, w_start+dx:w_end+dx]
            diff = torch.abs(source - target)
            total_tv += self.gamma * diff.mean()
        
        # 其他方向计算 (高惩罚)
        for dy, dx in self.other_directions:
            h_start = max(-dy, 0)
            h_end = H - max(dy, 0)
            w_start = max(-dx, 0)
            w_end = W - max(dx, 0)
            # 提取对比区域
            source = x[:, :, h_start:h_end, w_start:w_end]
            target = x[:, :, h_start+dy:h_end+dy, w_start+dx:w_end+dx]
            diff = torch.abs(source - target)
            total_tv += diff.mean()
            
        return self.weight * total_tv

    
def besselh(v, kind, z):
    import torch.special as sp
    if isinstance(z,np.float64):
        z = torch.Tensor([z])
    if v == 0:
        jv = sp.bessel_j0(z)
        yv = sp.bessel_y0(z)
    elif v == 1:
        jv = sp.bessel_j1(z)
        yv = sp.bessel_y1(z)
    else:
        raise NotImplementedError
    if kind == 1:
        return jv + 1j * yv
    elif kind == 2:
        return jv - 1j * yv
    else:
        raise NotImplementedError
    
def plot_J_figure(data, testsavedir_img, args):
    '''
    data --> 4096(pixel number) * transmitter number * 2(real and imag part)
    '''
    if not data.shape[-1] == 2:
        data = np.concatenate([np.real(data)[...,None], np.imag(data)[...,None]], axis=-1)
    reshaped_data = data.reshape(args.grid_num, args.grid_num, args.N_inc, 2)
    fig, axes = plt.subplots(2, args.N_inc, figsize=(25, 6))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    for i in range(args.N_inc):
        for j in range(2):
            ax = axes[j, i]
            sc = ax.imshow(reshaped_data[:, :, i, j].T, cmap='viridis')
            ax.axis('off')
            sc.set_cmap('jet')
    fig.colorbar(sc, cax=cbar_ax)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(testsavedir_img)
    plt.close()

def plot_J_figure_multifreqs(data, testsavedir_img, args):
    '''
    data --> 4096(pixel number) * transmitter number * 2(real and imag part)
    '''
    N_inc = args.N_inc_use if args.N_inc_use is not None else args.N_inc
    expected_size = len(args.freq) * args.grid_num * args.grid_num * N_inc * 2
    if data.size != expected_size:
        # plot bug for small N_inc; skip plotting silently
        return
    if not data.shape[-1] == 2:
        data = np.concatenate([np.real(data)[...,None], np.imag(data)[...,None]], axis=-1)
    reshaped_data = data.reshape(len(args.freq), args.grid_num, args.grid_num, N_inc, 2)
    fig, axes = plt.subplots(len(args.freq), N_inc, figsize=(25, 6))
    axes = np.asarray(axes).reshape((len(args.freq), N_inc))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    for i in range(N_inc):
        for j in range(len(args.freq)):
            ax = axes[j, i]
            sc = ax.imshow(np.linalg.norm(reshaped_data[j, :, :, i], axis=-1).T, cmap='viridis')
            ax.axis('off')
            sc.set_cmap('jet')
    fig.colorbar(sc, cax=cbar_ax)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(testsavedir_img)
    plt.close()

def plot_params_figure(epsilon, save_plot_path_params):
    if epsilon.ndim == 1 or (epsilon.ndim == 2 and epsilon.shape[1] == 1):
        epsilon = np.reshape(epsilon, (int(np.sqrt(epsilon.shape[0])), -1))
    if np.iscomplexobj(epsilon):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sc_real = axes[0].imshow(np.real(epsilon), cmap='viridis', origin='upper', aspect='auto')
        axes[0].set_title('Real(epsilon)')
        fig.colorbar(sc_real, ax=axes[0])
        sc_imag = axes[1].imshow(np.imag(epsilon), cmap='viridis', origin='upper', aspect='auto')
        axes[1].set_title('Imag(epsilon)')
        fig.colorbar(sc_imag, ax=axes[1])
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sc = ax.imshow(epsilon, cmap='viridis', origin='upper', aspect='auto')#, vmin=1.0, vmax=2.0)
        # sc.set_cmap('jet')
        fig.colorbar(sc, ax=ax)
    plt.savefig(save_plot_path_params, bbox_inches='tight', dpi=300)
    plt.close()

class LossRecorder:
    def __init__(self, save_path, loss_names=['total_loss', 'recon_loss', 'kl_loss']):
        self.save_path = save_path
        self.loss_names = loss_names
        self.history = {name: [] for name in loss_names}
        
    def update(self, current_losses):
        for name in self.loss_names:
            self.history[name].append(current_losses[name])
    
    def plot_losses(self, figure_name='loss_history.png'):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(self.history[self.loss_names[0]]) + 1)
        
        for name in self.loss_names:
            plt.plot(epochs_range, self.history[name], label=name, linewidth=2)
            
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('Loss', fontsize=14, fontweight='bold')
        plt.ylim([0,2])
        plt.title('Training Loss History', fontsize=16, fontweight='bold', pad=15)
        plt.legend(fontsize=12, frameon=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, figure_name), bbox_inches='tight', dpi=300)
        plt.close()
    
    def save_history(self, filename='loss_history.npy'):
        np.save(os.path.join(self.save_path, filename), self.history)
    
    def load_history(self, filename='loss_history.npy'):
        path = os.path.join(self.save_path, filename)
        if os.path.exists(path):
            self.history = np.load(path, allow_pickle=True).item()

def kl_divergence(init_latent):
    latent_mu = torch.mean(init_latent)    # 均值
    latent_logvar = torch.log(torch.var(init_latent) + 1e-8)  # 对数方差
    kl = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())
    return kl

import csv
def csv_writer(filename, header=None, data=None):
    """
    线程安全的CSV写入函数
    :param filename: CSV文件名
    :param header: 表头列表
    :param data: 要写入的数据列表
    """
    import os
    is_empty = not os.path.exists(filename) or os.path.getsize(filename) == 0
    
    # 需要写入表头的情况
    if is_empty:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(header)
    
    # 追加数据行
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data)
        
def read_exp_data(filename, nf, ns, nr):
    """
    读取实验数据文件
    对应MATLAB中的textread函数
    """
    from scipy.io import loadmat

    # 读取数据文件，跳过前10行
    data = np.loadtxt(filename, skiprows=10)
    
    # 提取散射场和入射场
    p_sca_truth = (data[:, 3] - data[:, 5]) + 1j * (data[:, 4] - data[:, 6])
    p_inc_truth = data[:, 5] + 1j * data[:, 6]
    p_sca_truth_bak = np.copy(p_sca_truth)
    
    # 加载校准系数
    # try:
    calib_file = os.path.join(os.path.dirname(filename), 'calib_rats_{}.mat'.format(filename.split('/')[-1][4:-6]))
    calib_data = loadmat(calib_file)
    calib_rats = calib_data['calib_rats'].flatten()
    
    # 应用校准系数 freq_n*trans_n
    # s0-f1f2f3f3, s1-f1f2f3f3
    for ii in range(len(data)):
        num_s = int(data[ii, 0])
        num_f = int(data[ii, 2]) - 1
        coef = calib_rats[(num_s - 1) * nf + num_f - 1]
        p_sca_truth[ii] = p_sca_truth[ii] / coef
        p_inc_truth[ii] = p_inc_truth[ii] / coef
        
    # 重塑数据
    p_sca_truth_mat = p_sca_truth.reshape(ns, nr, nf)
    p_sca_truth_mat = np.transpose(p_sca_truth_mat, (2, 0, 1))  # [freq][transmitter][receiver]
    p_sca_truth_bak_mat = p_sca_truth_bak.reshape(ns, nr, nf)
    p_sca_truth_bak_mat = np.transpose(p_sca_truth_bak_mat, (2, 0, 1))  # [freq][transmitter][receiver]
    
    return torch.tensor(p_sca_truth_mat, dtype=torch.complex64), torch.tensor(p_sca_truth_bak_mat, dtype=torch.complex64)
    # except:
    #     print("无法加载校准数据，使用原始数据")
    #     return torch.tensor(p_sca_truth, dtype=torch.complex64)
    
def fresnel_data_preprocess(nf, nr, trans_n, receiv_n, grid_num, device):
    # 获取掩码
    masks = data_in_need_from_vie(nf, nr, receiv_n, trans_n, device=device)
    # 创建垂直堆叠的掩码
    # 每个发射器对应的接收器数 (与MATLAB代码一致)
    masks_full = torch.zeros((trans_n, nr, receiv_n), device=device)
    for ii in range(trans_n):
        mask_ii = torch.zeros(nr, receiv_n, device=device)
        tmp1 = masks[ii, :]
        tmp = torch.nonzero(tmp1).squeeze()
        for hh in range(nr):
            mask_ii[hh, tmp[hh]] = 1
        masks_full[ii, ...] = mask_ii
    masks_full = masks_full.to(torch.complex64)
    return masks_full

def data_in_need_from_vie(nf, nr, receiv_n, trans_n, device='cuda'):
    """
    处理VIE的数据
    
    参数:
        e_test: 测试数据 [freq, trans_n, receiv_n]
        nf: 频率数量
        ns: 源数量
        receiv_n: 接收器数量
        trans_n: 发射器数量
        device: 计算设备
        
    返回:
        yout: 处理后的输出数据
        masks: 指示观测数据的掩码
    """
    masks = torch.zeros((trans_n, receiv_n), device=device)
    
    # 处理每个频率和源的数据
    for ii in range(nf):
        for jj in range(trans_n):
            
            # 计算无响应区域
            theta0 = (jj * 360) / trans_n
            theta_no_response = []
            for t in range(-59, 60):
                angle = (theta0 + t) % 360
                theta_no_response.append(int(angle))
            
            # 创建掩码
            mask = torch.ones((receiv_n), dtype=torch.float32, device=device)
            mask[theta_no_response] = 0
            
            # 保存第一个频率的掩码
            if ii == 0:
                masks[jj, :] = mask
    
    # 确保掩码是二值的
    masks = (masks != 0).float()
    
    return masks

def plot_multi_subfigures(data,root_path='./'):
    # 创建画布和8个子图，2行4列
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    fig.suptitle('8 Subplots Example', fontsize=16)

    # 循环绘制每个子图
    for i, ax in enumerate(axes.flat):
        for j,data_tmp in enumerate(data):
            # 绘制曲线
            ax.plot(data_tmp[i], linewidth=2, label=f'Data {j+1}')
        # 设置坐标轴范围
        ax.set_title(f'Plot {i+1}')
        
        # 设置坐标轴标签
        ax.set_xlabel('Receiv Index')
        ax.set_ylabel('Y axis')
        
        # 添加网格
        ax.grid(True)
    plt.legend()
    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为总标题留出空间
    plt.savefig(root_path+'8_subplots_example.png', dpi=300)
    
import pywt
def wden_pytorch(signal, wavelet='sym16', level=None, mode='soft'):
    # Auto-calculate decomposition level if not specified
    new_signal = np.zeros_like(signal)
    for i, signal_i in enumerate(signal):
        if level is None:
            level = pywt.dwt_max_level(len(signal_i), wavelet)
        
        # Wavelet decomposition
        coeffs = pywt.wavedec(signal_i, wavelet, level=level)
        
        # Thresholding (similar to 'sqtwolog' rule)
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal_i)))
        
        # Apply threshold
        coeffs[1:] = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
        
        # Reconstruct signal_i
        denoised = pywt.waverec(coeffs, wavelet)
        new_signal[i] = denoised[:len(signal_i)]
    
    return new_signal

def plot_arrays_row(arrays, save_path, titles=None, figsize=(15, 6), cmap='viridis', 
                   ylabel='Y axis', xlabel_list=None, vmin=None, vmax=None,
                   yticks_max=None, xticks_max=None, circles=None):
    font_size1 = 40
    font_size2 = 30
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = font_size1
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
    n_plots = len(arrays)
    figsize = (n_plots * (figsize[1]-0.5), figsize[1])  # Adjust width based on number of plots
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
    if n_plots == 1:
        axes = [axes]
    # Determine colorbar range if not provided
    if vmin is None:
        vmin = min(arr.min() for arr in arrays)
    if vmax is None:
        vmax = max(arr.max() for arr in arrays)
    ims = []
    for i, (ax, arr) in enumerate(zip(axes, arrays)):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ims.append(im)
        if titles:
            ax.set_title(titles[i])
        if xlabel_list:
            if isinstance(xlabel_list, str):
                ax.set_xlabel(f'${xlabel_list}$', fontsize=font_size1)
            else:
                ax.set_xlabel(f'${xlabel_list[i]}$', fontsize=font_size1)
        # Only show y-label on leftmost plot
        if i == 0:
            ax.set_ylabel(f'${ylabel}$', fontsize=font_size1)
        # Custom ticks
        if xticks_max is not None:
            ticks = np.arange(0, len(arrays[0]), len(arrays[0])//3)
            ax.set_xticks(ticks)
            xticklabels = [f'{x * xticks_max / len(arrays[0]):.2f}' for x in ticks]
            ax.set_xticklabels(xticklabels)
        if yticks_max is not None:
            ticks = np.arange(0, len(arrays[0]), len(arrays[0])//3)  # 设置5个刻度点
            ax.set_yticks(ticks)
            yticklabels = [f'{yticks_max-(y * yticks_max / len(arrays[0])):.2f}' for y in ticks]
            ax.set_yticklabels(yticklabels)

        ax.tick_params(axis='both', which='major', labelsize=font_size2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            
        # 绘制圆形
        if circles:
            for (center_x, center_y), radius in circles:
                # 将物理坐标转换为像素坐标
                if xticks_max is not None and yticks_max is not None:
                    pixel_center_x = center_x * len(arr) / xticks_max
                    pixel_center_y = (yticks_max - center_y) * len(arr) / yticks_max
                    pixel_radius = radius * len(arr) / xticks_max
                    
                    circle = plt.Circle((pixel_center_x, pixel_center_y), pixel_radius, 
                                       fill=False, linestyle='--', color='red', linewidth=2)
                    ax.add_patch(circle)
    
    # Add shared colorbar
    plt.tight_layout(rect=[0, 0, 0.95, 1], pad=0.1, w_pad=0.2)
    cax = fig.add_axes([0.96, 0.18, 0.01, 0.73])  # [left, bottom, width, height]
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.ax.tick_params(labelsize=font_size2)
    cbar.ax.set_xlabel(r'$\boldsymbol{\chi}$', fontsize=font_size2+5, labelpad=2, fontfamily='Times New Roman')
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_arrays_column(arrays, save_path, titles=None, figsize=(6, 20), cmap='viridis', 
                   ylabel='Y axis', xlabel_list=None, vmin=None, vmax=None,
                   yticks_max=None, xticks_max=None, circles=None):
    font_size1 = 40
    font_size2 = 30
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = font_size1
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
    n_plots = len(arrays)
    figsize = (figsize[0], n_plots * (figsize[0]-0.5))  # Adjust width based on number of plots
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharey=True)
    if n_plots == 1:
        axes = [axes]
    # Determine colorbar range if not provided
    if vmin is None:
        vmin = min(arr.min() for arr in arrays)
    if vmax is None:
        vmax = max(arr.max() for arr in arrays)
    ims = []
    for i, (ax, arr) in enumerate(zip(axes, arrays)):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ims.append(im)
        if titles:
            ax.set_title(titles[i])
        if xlabel_list:
            if isinstance(xlabel_list, str):
                ax.set_xlabel(f'${xlabel_list}$', fontsize=font_size1)
            else:
                ax.set_xlabel(f'${xlabel_list[i]}$', fontsize=font_size1)
        # Only show y-label on leftmost plot
        if i == 0:
            ax.set_ylabel(f'${ylabel}$', fontsize=font_size1)
        # Custom ticks
        if xticks_max is not None:
            ticks = np.arange(0, len(arrays[0]), len(arrays[0])//3)
            ax.set_xticks(ticks)
            xticklabels = [f'{x * xticks_max / len(arrays[0]):.2f}' for x in ticks]
            ax.set_xticklabels(xticklabels)
        if yticks_max is not None:
            ticks = np.arange(0, len(arrays[0]), len(arrays[0])//3)  # 设置5个刻度点
            ax.set_yticks(ticks)
            yticklabels = [f'{yticks_max-(y * yticks_max / len(arrays[0])):.2f}' for y in ticks]
            ax.set_yticklabels(yticklabels)

        ax.tick_params(axis='both', which='major', labelsize=font_size2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            
        # 绘制圆形
        if circles:
            for (center_x, center_y), radius in circles:
                # 将物理坐标转换为像素坐标
                if xticks_max is not None and yticks_max is not None:
                    pixel_center_x = center_x * len(arr) / xticks_max
                    pixel_center_y = (yticks_max - center_y) * len(arr) / yticks_max
                    pixel_radius = radius * len(arr) / xticks_max
                    
                    circle = plt.Circle((pixel_center_x, pixel_center_y), pixel_radius, 
                                       fill=False, linestyle='--', color='red', linewidth=2)
                    ax.add_patch(circle)
    
    # Add shared colorbar
    plt.tight_layout(rect=[0, 0, 0.95, 1], pad=0.1, w_pad=0.2)
    cax = fig.add_axes([0.96, 0.18, 0.01, 0.73])  # [left, bottom, width, height]
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.ax.tick_params(labelsize=font_size2)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_single_figure(epsilon, save_plot_path_params=None, 
                       ylabel='Y axis', xlabel='X axis',
                       cmap='viridis', vmin=None, vmax=None,
                       xticks_max=None, yticks_max=None, circles=None):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 26
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
    if epsilon.shape[1] == 1:
        epsilon = np.reshape(epsilon, (int(np.sqrt(epsilon.shape[0])), -1))
    if vmin is None:
        vmin = np.min(epsilon)
    if vmax is None:
        vmax = np.max(epsilon)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    sc = ax.imshow(epsilon, cmap='viridis', origin='upper', aspect='equal', vmin=vmin, vmax=vmax)
    if xticks_max is not None:
        ticks = np.arange(0, len(epsilon), len(epsilon)//3)
        ax.set_xticks(ticks)
        xticklabels = [f'{x * xticks_max / len(epsilon):.2f}' for x in ticks]
        ax.set_xticklabels(xticklabels)
    if yticks_max is not None:
        ticks = np.arange(0, len(epsilon), len(epsilon)//3)  # 设置5个刻度点
        ax.set_yticks(ticks)
        yticklabels = [f'{yticks_max-(y * yticks_max / len(epsilon)):.2f}' for y in ticks]
        ax.set_yticklabels(yticklabels)
    if circles:
        for (center_x, center_y), radius in circles:
            # 将物理坐标转换为像素坐标
            if xticks_max is not None and yticks_max is not None:
                pixel_center_x = center_x * len(epsilon) / xticks_max
                pixel_center_y = (yticks_max - center_y) * len(epsilon) / yticks_max
                pixel_radius = radius * len(epsilon) / xticks_max
                
                circle = plt.Circle((pixel_center_x, pixel_center_y), pixel_radius, 
                                    fill=False, linestyle='--', color='red', linewidth=2)
                ax.add_patch(circle)
    sc.set_cmap(cmap)
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=22)
    plt.xlabel(f'${xlabel}$')
    plt.ylabel(f'${ylabel}$')
    ax.tick_params(axis='both', which='major', labelsize=22)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    plt.tight_layout()
    plt.savefig(save_plot_path_params, bbox_inches='tight', dpi=300)
    plt.close()

def plot_loss_curve(history, save_path):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.figure(figsize=(10, 6))
    loss_names = list(history.keys())
    epochs_range = range(1, len(history[loss_names[0]]) + 1)
    
    for name in loss_names:
        plt.plot(epochs_range, history[name], label=name, linewidth=2)
        
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.ylim([0,2])
    plt.title('Training Loss History', fontsize=16, fontweight='bold', pad=15)
    plt.legend(fontsize=12, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def read_deepcsi_data(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    data['J'] = data['J_pred']
    return data

def read_csi_data(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    data['J'] = data['omega']
    data['epsilon'] = np.flipud(data['epsilon'].real.reshape(int(np.sqrt(data['epsilon'].shape[0])), -1, order='F'))
    return data

def plot_loss_curve(history, save_path):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.figure(figsize=(8, 6))
    loss_names = list(history.keys())
    epochs_range = range(1, len(history[loss_names[0]]) + 1)
    
    markers = ['o', 's', '^', 'D']
    for i,name in enumerate(loss_names):
        plt.plot(epochs_range, history[name], label=name, 
                 linewidth=3, marker=markers[i],markevery=10, markersize=8)
        
    plt.xlabel('Epoch', fontsize=24, fontweight='bold')
    plt.ylabel('Loss', fontsize=24, fontweight='bold')
    plt.ylim([0,2])
    # plt.title('Training Loss History', fontsize=28, fontweight='bold', pad=15)
    plt.legend(fontsize=24, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    x_ticks = np.linspace(0, len(epochs_range), 6, dtype=int)  # 设置6个刻度点
    plt.xticks(x_ticks, x_ticks, fontsize=24)
    y_ticks = np.linspace(0, 2, 5)  # 设置5个刻度点：0.0, 0.5, 1.0, 1.5, 2.0
    plt.yticks(y_ticks, fontsize=24)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def read_loss_data(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    key_mapping = {
        'J_state_loss': 'StateLoss',
        'Esca_loss': 'DataLoss',
        'TV_loss': 'TVLoss',
        'BP_loss': 'TotalLoss'
    }
    # 创建新字典并重命名键
    renamed_data = {key_mapping[k]: v for k, v in data.items()}
    return renamed_data

def plot_misfit_curve_twin(history_left, history_right, x_list, save_path, ylabel_left='RMSE', ylabel_right='SSIM'):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    fig, ax1 = plt.subplots(figsize=(10, 7))
    # 设置左y轴
    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (name, values) in enumerate(history_left.items()):
        ax1.plot(x_list, values, label=f'$\\mathrm{{{ylabel_left}}}_{{\\mathrm{{{name}}}}}$', color=colors[i],
                linewidth=2, marker=markers[i], markevery=1, markersize=6)
    
    ax1.set_xlabel('Noise Level', fontsize=24, fontweight='bold')
    ax1.set_ylabel(ylabel_left, fontsize=24, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # 设置右y轴
    ax2 = ax1.twinx()
    for i, (name, values) in enumerate(history_right.items()):
        ax2.plot(x_list, values, label=f'$\\mathrm{{{ylabel_right}}}_{{\\mathrm{{{name}}}}}$', color=colors[i],
                linewidth=2, marker=markers[i], markevery=1, 
                markersize=6, linestyle='--')
    
    ax2.set_ylabel(ylabel_right, fontsize=24, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # 设置网格和刻度
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    
    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              fontsize=18, frameon=True, loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_misfit_curve(history_left, history_right, x_list, save_path, ylabel_left='RMSE', ylabel_right='SSIM'):
    """
    绘制RMSE和SSIM指标对比曲线，共享y轴
    
    Args:
        history_left (dict): RMSE数据，key为方法名，value为数值list
        history_right (dict): SSIM数据，key为方法名，value为数值list
        x_list (list): x轴数据点
        save_path (str): 图像保存路径
        ylabel_left (str): 第一个指标名称
        ylabel_right (str): 第二个指标名称
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    fig, ax = plt.subplots(figsize=(13, 9))
    
    # 定义样式
    markers = ['o', 's', '^', 'D', 'v', 'h']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 绘制第一个指标 (RMSE)
    for i, (name, values) in enumerate(history_left.items()):
        ax.plot(x_list, values, 
                label=f'$\\mathrm{{{ylabel_left}}}_{{\\mathrm{{{name}}}}}$', 
                color=colors[i], linewidth=3, 
                marker=markers[i], markevery=1, markersize=10)
    
    # 绘制第二个指标 (SSIM)，使用虚线
    for i, (name, values) in enumerate(history_right.items()):
        ax.plot(x_list, values, 
                label=f'$\\mathrm{{{ylabel_right}}}_{{\\mathrm{{{name}}}}}$', 
                color=colors[i], 
                linewidth=3, linestyle='--', 
                marker=markers[i], markevery=1, markersize=10)
    
    # 设置坐标轴
    ax.set_xlabel('Noise Level', fontsize=26)
    ax.set_ylabel('Performance Metrics', fontsize=26)
    
    # 设置网格和刻度
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(labelsize=20)
    
    # 设置图例
    ax.legend(fontsize=20, frameon=True, loc='best', ncol=2, columnspacing=0.5, handletextpad=0.5, borderpad=0.4)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
def plot_losscomp_curve(history, save_path):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 26
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.figure(figsize=(7.5, 6))
    method_list = list(history.keys())
    loss_names = list(history[method_list[0]].keys())
    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    line_styles = ['-', ':', '-.', ':']
    
    for nm_idx, name in enumerate(loss_names):
        for mtd_idx, method in enumerate(method_list):
            epochs_range = range(1, len(history[method][name]) + 1)
            plt.plot(epochs_range, history[method][name], label=f'${{\\mathrm{{{name}}}}}-\\mathrm{{{method}}}$', linewidth=2,
                     marker=markers[mtd_idx], linestyle=line_styles[nm_idx], markevery=5)
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([0,2])
    plt.legend(fontsize=22, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
def plot_scatter_field_comparison(data_dict, x_values=None, save_path=None, 
                                 title=None, labels=None,
                                 title_real='Real Part',
                                 title_imag='Imaginary Part',
                                 xlabel='Receiver Index',
                                 ylabel='Amplitude'):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 26
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), constrained_layout=True)
    # plt.subplots_adjust(wspace=0.01)
    
    # 定义样式
    markers = ['o', 's', '^', 'D', 'v', 'h']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--', ':', '-.']
    
    # 如果没有提供标签，使用方法名
    if labels is None:
        labels = {method: method for method in data_dict.keys()}
    
    # 如果没有提供x坐标，使用索引
    if x_values is None:
        # 假设所有数组长度相同，获取第一个数组的长度
        first_key = list(data_dict.keys())[0]
        x_values = np.arange(len(data_dict[first_key]))
    
    # 绘制实部（左图）
    for i, (method, data) in enumerate(data_dict.items()):
        marker_index = i % len(markers)
        color_index = i % len(colors)
        style_index = i % len(line_styles)
        
        # 提取实部
        real_data = np.real(data)
        ax1.plot(x_values, real_data, 
                label=labels[method],
                color=colors[color_index], 
                linestyle=line_styles[style_index], 
                linewidth=2.5, 
                markevery=max(1, len(x_values)//20),  # 每20个点标记一次
                markersize=8)
    
    # 绘制虚部（右图）
    for i, (method, data) in enumerate(data_dict.items()):
        marker_index = i % len(markers)
        color_index = i % len(colors)
        style_index = i % len(line_styles)
        
        # 提取虚部
        imag_data = np.imag(data)
        ax2.plot(x_values, imag_data, 
                label=labels[method],
                color=colors[color_index], 
                linestyle=line_styles[style_index], 
                linewidth=2.5, 
                markevery=max(1, len(x_values)//20),  # 每20个点标记一次
                markersize=8)
    
    # 设置标题和坐标轴标签
    if title:
        fig.suptitle(title, fontsize=28, fontweight='bold')
    ax1.set_xlabel(f'$\\mathrm{{{xlabel}}}$', fontsize=26, fontweight='bold')
    ax1.set_ylabel(f'$E_{{sca}}$ $\\mathrm{{Real}}$', fontsize=26, fontweight='bold')
    
    ax2.set_xlabel(f'$\\mathrm{{{xlabel}}}$', fontsize=26, fontweight='bold')
    ax2.set_ylabel(f'$E_{{sca}}$ $\\mathrm{{Imag}}$', fontsize=26, fontweight='bold')
    
    # 设置网格和刻度
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(labelsize=22)
    ax2.tick_params(labelsize=22)
    
    # 设置图例
    ax1.legend(fontsize=22, frameon=True, loc='upper right', ncol=2, columnspacing=1.0)
    ax2.legend(fontsize=22, frameon=True, loc='upper right', ncol=2, columnspacing=1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
def plot_scatter_field_comparison_new(data_dict, x_values=None, save_path=None, 
                                 title=None, labels=None,
                                 title_real='Real Part',
                                 title_imag='Imaginary Part',
                                 xlabel='Receiver Index',
                                 ylabel='Amplitude'):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 32
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 7), constrained_layout=True)
    # plt.subplots_adjust(wspace=0.01)
    
    # 定义样式
    markers = ['o', 's', '^', 'D', 'v', 'h', '<', '>', 'x', '+', '*']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--']
    
    # 如果没有提供标签，使用方法名
    if labels is None:
        labels = {method: method for method in data_dict.keys()}
    
    # 如果没有提供x坐标，使用索引
    if x_values is None:
        # 假设所有数组长度相同，获取第一个数组的长度
        first_key = list(data_dict.keys())[0]
        x_values = np.arange(len(data_dict[first_key]))
    
    # 绘制实部（左图）
    for i, (method, data) in enumerate(data_dict.items()):
        marker_index = i % len(markers)
        color_index = i % len(colors)
        style_index = 0 if i==0 else 1
        
        # 提取实部
        real_data = np.real(data)
        ax1.plot(x_values, real_data, 
                label=labels[method],
                color=colors[color_index], 
                linestyle=line_styles[style_index], 
                marker=markers[marker_index],
                linewidth=3.5, 
                markevery=max(1, 20),  # 每20个点标记一次
                markersize=8)
    
    # 绘制虚部（右图）
    for i, (method, data) in enumerate(data_dict.items()):
        marker_index = i % len(markers)
        color_index = i % len(colors)
        style_index = 0 if i==0 else 1
        
        # 提取虚部
        imag_data = np.imag(data)
        ax2.plot(x_values, imag_data, 
                label=labels[method],
                color=colors[color_index], 
                linestyle=line_styles[style_index], 
                marker=markers[marker_index],
                linewidth=3.5, 
                markevery=max(1, 20),  # 每20个点标记一次
                markersize=8)
    
    # 设置标题和坐标轴标签
    if title:
        fig.suptitle(title, fontsize=28, fontweight='bold')
    ax1.set_xlabel(f'$\\mathrm{{{xlabel}}}$', fontsize=32, fontweight='bold')
    ax1.set_ylabel(f'$E_{{sca}}$ $\\mathrm{{Real}}$', fontsize=32, fontweight='bold')
    
    ax2.set_xlabel(f'$\\mathrm{{{xlabel}}}$', fontsize=32, fontweight='bold')
    ax2.set_ylabel(f'$E_{{sca}}$ $\\mathrm{{Imag}}$', fontsize=32, fontweight='bold')
    
    # 设置网格和刻度
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(labelsize=30)
    ax2.tick_params(labelsize=30)
    
    # 设置图例
    ax1.legend(fontsize=30, frameon=True, loc='upper right', ncol=2, columnspacing=1.0)
    ax2.legend(fontsize=30, frameon=True, loc='upper right', ncol=2, columnspacing=1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

class SSIM_Metric():
    """自定义 SSIM 模块，避免每次调用时重复构造。"""

    def __init__(self, data_range: float = 1.0) -> None:
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

    def get_ssim_metric(self, pred_eps, gt_eps):
        pred_tensor = torch.from_numpy(pred_eps).float()
        gt_tensor = torch.from_numpy(gt_eps).float()
        return  self.ssim(pred_tensor.unsqueeze(0).unsqueeze(0),
                gt_tensor.unsqueeze(0).unsqueeze(0)).item()

def get_mse_metric(pred_eps, gt_eps):
    pred_tensor = torch.from_numpy(pred_eps).float()
    gt_tensor = torch.from_numpy(gt_eps).float()
    rmse = torch.sqrt(F.mse_loss(pred_tensor, gt_tensor) / torch.mean(gt_tensor ** 2)).item()
    return rmse

def format_print(ssim_values, mse_values, method_list, diff_varies_list):
    method_short_names = {
        'params': 'Params',
        'paramzation': 'Paramz',
        'proj': 'Projjj',
        'MRCSI': 'MRCSI',
    }
    print("SSIM Values:")
    headers = ['Method'] + [f'{varies}' for varies in diff_varies_list]
    table_data = [
        [method_short_names[m.split('_')[0]]] + [f'{v:.6f}' for v in ssim_values[m]]
        for m in method_list
    ]
    print(tabulate(table_data, headers=headers, tablefmt='simple', stralign='right'))
    print("MSE Values:")
    table_data = [
        [method_short_names[m.split('_')[0]]] + [f'{v:.6f}' for v in mse_values[m]]
        for m in method_list
    ]
    print(tabulate(table_data, headers=headers, tablefmt='simple', stralign='right'))

def plot_ssim_rmse_trends(noise_levels, ssim_values, rmse_values, method_list, 
                          shape_name, save_path, method_labels=None):
    """
    Plot SSIM and RMSE trends versus noise level for multiple methods.

    Args:
        noise_levels (list[float]): Noise levels in the same order as the metrics.
        ssim_values (dict[str, list[float]]): SSIM values keyed by method name.
        rmse_values (dict[str, list[float]]): RMSE values keyed by method name.
        method_list (list[str]): Ordering of methods to display.
        shape_name (str): Shape identifier for titles.
        save_path (str): Output path for the figure.
        method_labels (dict[str, str], optional): Custom labels for legend.
    """
    default_labels = {
        'MRCSI': 'MRCSI',
        'params': 'DeepCSI',
        'paramzation_sdvaeft_latent256': 'CasVAE',
        'proj_param200_sdvaeft_latent256': 'LaMP',
    }
    label_map = method_labels or default_labels
    palette = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
    markers = ['o', 's', '^', 'D', 'v', '>']

    def _filter_mrcsi_outliers(noises, ssim_series, rmse_series):
        """
        Drop MRCSI points where SSIM is very low and RMSE very high.
        Keeps raw values for all other methods and MRCSI points.
        """
        filtered = [
            (n, s, r)
            for n, s, r in zip(noises, ssim_series, rmse_series)
            if not (s < 0.03 or r > 1.0)
        ]
        if not filtered:
            return noises, ssim_series, rmse_series
        n_keep, s_keep, r_keep = zip(*filtered)
        return list(n_keep), list(s_keep), list(r_keep)

    with plt.rc_context({
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix',
        'axes.edgecolor': '#3a3a3a',
        'axes.linewidth': 1.25,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    }):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        plotted_rmse_max = 0.0

        for idx, method in enumerate(method_list):
            color = palette[idx % len(palette)]
            marker = markers[idx % len(markers)]
            label = label_map.get(method, label_map.get(method.split('_')[0], method))

            ssim_series = ssim_values.get(method, [])
            rmse_series = rmse_values.get(method, [])
            noise_use = noise_levels
            if method == 'MRCSI':
                noise_use, ssim_series, rmse_series = _filter_mrcsi_outliers(
                    noise_levels, ssim_series, rmse_series
                )
            if len(ssim_series) != len(noise_use) or len(rmse_series) != len(noise_use):
                print(f"Warning: metric length mismatch for {method} on {shape_name}, skipping plot.")
                continue
            if rmse_series:
                plotted_rmse_max = max(plotted_rmse_max, max(rmse_series))

            axes[0].plot(
                noise_use, ssim_series, label=label,
                color=color, marker=marker, markersize=7.5,
                linewidth=2.8, alpha=0.95)
            axes[1].plot(
                noise_use, rmse_series, label=label,
                color=color, marker=marker, markersize=7.5,
                linewidth=2.8, alpha=0.95)

        axes[0].set_ylabel('SSIM', fontsize=18)
        axes[1].set_ylabel('RMSE', fontsize=18)
        axes[1].yaxis.set_label_position("right")
        axes[1].yaxis.tick_right()

        for ax in axes:
            ax.set_xlabel('Noise Level', fontsize=18)
            ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)
            ax.tick_params(axis='both', labelsize=14, length=6, width=1.1)
            ax.set_facecolor('#f7f7f7')
            for spine in ax.spines.values():
                spine.set_linewidth(1.1)

        axes[0].set_ylim(0, 1.02)
        if plotted_rmse_max > 0:
            axes[1].set_ylim(0, plotted_rmse_max * 1.08)

        handles, labels = axes[0].get_legend_handles_labels()
        legend_cols = min(len(handles), 4) if handles else 1
        fig.legend(handles, labels, ncol=legend_cols, frameon=False, fontsize=16,
                   loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   columnspacing=1.0, handlelength=2.5)

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
