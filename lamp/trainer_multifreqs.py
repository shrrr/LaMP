import os
import torch
import numpy as np
from time import time
import torch.nn as nn
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.special as sp
import lpips

from .utils import *
from .models import *
from .vae import *

class VAEProjector():
    def __init__(self, vae_decoder, latent_code, args, max_steps=50, loss_fn='rmsetv'):
        self.vae_decoder = vae_decoder
        self.latent_code = latent_code
        self.args = args
        self.max_steps = int(getattr(args, 'proj_inner_max', max_steps))
        self.min_steps = int(getattr(args, 'proj_inner_min', 0))
        self.rmse_tol = float(getattr(args, 'proj_inner_tol', 1.0e-2))
        self.z_init_kind = getattr(args, 'proj_z_init', 'ones')
        self.z_init_alpha = getattr(args, 'proj_z_init_alpha', None)
        self.z_init_ood_tau = float(getattr(args, 'proj_z_init_ood_tau', 0.15))
        # OOD score metric for safe_encode (decides how much z0 leans to ones vs
        # encode(eps)). 'meanL1_both' = original mean(|eps-clip|) (dilutes sparse
        # but severe overflow); 'rms_up' = sqrt(mean(relu(eps-max)^2)), an
        # upper-overflow L2 measure that is not diluted by background pixels and
        # ignores the benign eps<1 side.
        self.ood_metric = getattr(args, 'proj_ood_metric', 'meanL1_both')
        # When proj_tv_weight is exactly 0, fall back to absolute-MSE inner loss
        # (matches the Nov 2025 historical VAEProjector that hardcoded
        # loss_fn='mse'). The rmsetv mode normalizes the fidelity term by
        # mean(eps^2), which changes the effective Adam step size and prevents
        # bit-exact reproduction of the paper Fig 6 numbers.
        if float(getattr(args, 'proj_tv_weight', 0.05)) == 0.0:
            self.loss_fn = 'mse'
        else:
            self.loss_fn = loss_fn
        self.device = latent_code.device

        self.vae_decoder.eval()
        for p in self.vae_decoder.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(params=[self.latent_code], lr=args.params_lrate, betas=(0.9, 0.999))
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_loss_fn.eval()
        regular2func = {'tv_l1': L_TV_L1, 'tv_l2': L_TV_L2, 'mdtv_l1': MultiDirectionalTV}
        self.tv_fn = L_TV_L1()
        # Per-projection TV weight (0 disables the TV term inside Eq. (10);
        # used by the R2-1a ablation in the TAP revision).
        self.tv_weight = float(getattr(args, 'proj_tv_weight', 0.05))
        # Pre-compute the base latent code for the chosen --proj_z_init mode.
        # encode = encode(intermediate epsilon) at the first project() call.
        # ones / zeros / zero = compute encode of a constant contrast map
        # (all-ones / all-zeros) so that each projection event starts from the
        # same deterministic latent regardless of the upstream method.
        self._z_base = None
        if self.z_init_kind in ('ones', 'zeros', 'zero'):
            const = 1.0 if self.z_init_kind == 'ones' else 0.0
            ref = torch.full_like(latent_code.detach(), 0)  # placeholder; resolve in project()
            self._z_init_const = const
        self.globel_init = True

    def _resolve_init_latent(self, epsilon):
        with torch.no_grad():
            if self.z_init_kind == 'encode':
                z = self.vae_decoder.get_latent(epsilon.unsqueeze(0).unsqueeze(0))
            elif self.z_init_kind in ('ones', 'zeros', 'zero'):
                const = 1.0 if self.z_init_kind == 'ones' else 0.0
                ref = torch.full_like(epsilon, const)
                z = self.vae_decoder.get_latent(ref.unsqueeze(0).unsqueeze(0))
            elif self.z_init_kind == 'safe_encode':
                clip_min = 1.0
                clip_max = float(getattr(self.args, 'max_params', 2.0))
                eps_clip = epsilon.clamp(min=clip_min, max=clip_max)
                z_clip = self.vae_decoder.get_latent(eps_clip.unsqueeze(0).unsqueeze(0))
                z_ones = self.vae_decoder.get_latent(torch.ones_like(epsilon).unsqueeze(0).unsqueeze(0))
                span = max(clip_max - clip_min, 1.0e-6)
                if self.ood_metric == 'rms_up':
                    over_up = torch.relu(epsilon - clip_max)
                    ood_score = torch.sqrt(torch.mean(over_up**2)) / span
                elif self.ood_metric == 'mean_sigma':
                    # upper-overflow mean + std: the std term rescues sparse but
                    # severe overflow (e.g. a few center pixels at 8x clip) that a
                    # plain mean dilutes against the background.
                    over_up = torch.relu(epsilon - clip_max)
                    k = float(getattr(self.args, 'proj_ood_sigma_k', 1.0))
                    ood_score = (torch.mean(over_up) + k * torch.std(over_up)) / span
                else:  # 'meanL1_both' (original)
                    ood_score = torch.mean(torch.abs(epsilon - eps_clip)) / span
                tau = max(self.z_init_ood_tau, 1.0e-6)
                w = torch.exp(-ood_score / tau).clamp(0.0, 1.0)
                z = w * z_clip + (1.0 - w) * z_ones
            else:
                raise ValueError(f"unknown proj_z_init={self.z_init_kind}")
            if self.z_init_alpha is not None:
                z_gt = self.vae_decoder.get_latent(epsilon.unsqueeze(0).unsqueeze(0))
                a = float(self.z_init_alpha)
                z = a * z_gt + (1.0 - a) * z
            return z

    def project(self, epsilon, threadshold=0.005):
        """Project epsilon to VAE manifold; returns blended epsilon."""
        pbar = tqdm(range(self.max_steps), desc='Project into VAE latent space ...', leave=True)
        i = 0
        loss = torch.inf
        rmse = torch.inf
        if self.globel_init:
            with torch.no_grad():
                self.latent_code.copy_(self._resolve_init_latent(epsilon))
            self.globel_init = False
        while (i < self.min_steps or rmse > self.rmse_tol) and i < self.max_steps:
            self.optimizer.zero_grad()
            epsilon_pred = self.vae_decoder(self.latent_code)
            rmse = torch.sqrt(torch.mean((epsilon_pred - epsilon)**2)/torch.mean((epsilon)**2))
            if self.loss_fn == 'mse':
                loss = torch.mean((epsilon_pred - epsilon)**2)
            elif self.loss_fn == 'rmsetv':
                fidelity = torch.mean((epsilon_pred - epsilon)**2)/torch.mean((epsilon)**2)
                loss = fidelity + self.tv_weight * self.tv_fn(epsilon_pred)
            elif self.loss_fn == 'rmse':
                loss = torch.sqrt(torch.mean((epsilon_pred - epsilon)**2)/torch.mean((epsilon)**2))
            loss.backward()
            self.optimizer.step()
            pbar.set_description("Projecting | %d/%d | train_loss: %.2e | " % (i, self.max_steps, loss.item()))
            pbar.update(1)
            i += 1
        print('Projection done: rmse loss = %.4e (tv_weight=%.4f, inner_steps=%d)' % (rmse.item(), self.tv_weight, i))
        with torch.no_grad():
            epsilon.copy_(epsilon_pred.detach())
        return epsilon


class TVProjector():
    """Replaces the VAE projection step with a TV-only proximal optimization on
    the pixel-domain contrast estimate. Used by the R2-1a ablation (variant
    iv) in the TAP revision: same alternating framework as LaMP but with the
    learned manifold projection replaced by a periodic TV-prox step.

    The inner optimization solves
        argmin_chi  0.5 * ||chi - chi_hat||_2^2 / ||chi_hat||_2^2 + lambda * TV(chi)
    by `inner_steps` Adam updates on a freshly-cloned copy of `chi_hat`.
    Defaults (lambda=0.05, inner_steps=300, lr=1e-1, TV-L1) mirror the
    VAEProjector settings so that variant (iv) differs from the proposed
    method only in the prior, not in the optimizer schedule.
    """

    def __init__(self, args, inner_steps=300, tv_weight=0.05, inner_lr=1e-1):
        self.args = args
        self.inner_steps = int(getattr(args, 'proj_inner_steps', inner_steps))
        self.tv_weight = float(getattr(args, 'proj_tv_weight', tv_weight))
        self.inner_lr = float(getattr(args, 'proj_inner_lr', inner_lr))
        self.tv_fn = L_TV_L1()

    def project(self, epsilon, threadshold=0.005):
        target = epsilon.detach().clone()
        denom = torch.mean(target ** 2).clamp_min(1e-12)
        x = target.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=self.inner_lr, betas=(0.9, 0.999))
        pbar = tqdm(range(self.inner_steps), desc='Project via TV-prox ...', leave=True)
        last_loss = torch.inf
        for i in range(self.inner_steps):
            optimizer.zero_grad()
            fidelity = torch.mean((x - target) ** 2) / denom
            tv = self.tv_fn(x)
            loss = fidelity + self.tv_weight * tv
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
            pbar.set_description('TV-prox | %d/%d | loss %.2e' % (i, self.inner_steps, last_loss))
            pbar.update(1)
        print('TV-prox done: loss = %.4e (tv_weight=%.4f, inner_steps=%d)' % (last_loss, self.tv_weight, self.inner_steps))
        with torch.no_grad():
            epsilon.copy_(x.detach())
        return epsilon



class PINNsFwdIsp2D():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        
        self.eta_0 = 120 * np.pi
        c = 3e8
        self.eps_0 = 8.85e-12
        args.freq = torch.tensor([float(f) for f in args.freq.split(',') if not f==''])
        self.num_freqs = len(args.freq)
        self.lam_0 = c / (args.freq * 1e9)
        self.k_0 = 2 * np.pi / self.lam_0
        self.omega = self.k_0 * c

        self.step_size = args.L_doi / (args.grid_num - 1)
        self.cell_area = self.step_size ** 2
        self.a_eqv = np.sqrt(self.cell_area/np.pi)
        self.green_asembly(args)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def fwd_model(self, inputs, networkfwd_fn, embed_fn):
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = embed_fn(inputs_flat)
        outputs = None
        for networkfwd_fn_tmp in networkfwd_fn:
            outputs_flat = networkfwd_fn_tmp(embedded)
            if self.args.J_network == 'multi-branch':
                output_tmp = torch.reshape(outputs_flat, [self.num_freqs]+list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
            else:
                output_tmp = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
            outputs = output_tmp if outputs is None else torch.cat((outputs, output_tmp), 0)
        return outputs
    
    def create_isp_neuropretor(self, args):
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        skips = [4,8,12,16,20,24,28,32,36,40,44,48]
        # skips = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,40,44,48]
        # skips = [6,12,18,24,30,36,48]
        # skips = [8,16,24]
        # [3,6,9,12,14]
        if len(args.freq)>1 and args.J_network == 'multi-mlp':
            model_J = []
            grad_vars = []
            for _ in args.freq:
                model_J.append(NeJF(D=args.netdepth, W=args.netwidth,
                                    input_ch=int(input_ch * 2), output_ch=2, skips=skips, tanh=True).to(self.device))
                grad_vars.append(model_J[-1].parameters())
        elif args.J_network == 'multi-branch':
            model_J = [MultiFrequencyNet(layer_num=args.netdepth, num_freqs=self.num_freqs, input_dim=int(input_ch * 2), 
                                         output_dim=2, shared_dim=args.netwidth, skips=skips, learnable_tanh=False, inc_num=args.N_inc).to(self.device)]
            grad_vars = list(model_J[0].parameters())
        elif args.J_network == 'identify':
            init_J = torch.zeros(self.num_freqs, self.args.grid_num**2, self.args.N_inc, 2)
            model_J = nn.Parameter(init_J.to(self.device), requires_grad=True)
            grad_vars = [model_J]
        elif args.J_network == 'single-mlp':
            model_J = [NeJF(D=args.netdepth, W=args.netwidth,
                        input_ch=int(input_ch * 2.5) if len(args.freq)>1 else (input_ch * 2), output_ch=2, skips=skips, tanh=True).to(self.device)]
            grad_vars = list(model_J[0].parameters())
        model_fwd_fn = lambda inputs, networkfwd_fn: self.fwd_model(inputs, networkfwd_fn, embed_fn=embed_fn) if not isinstance(networkfwd_fn, torch.Tensor) else networkfwd_fn
        if args.epsilon_network == 'mlp':
            model_epsilon = NeJF(D=args.netdepth, W=args.netwidth,
                                input_ch=input_ch, output_ch=1, skips=skips, tanh=True, epsilon=True).to(self.device)
            grad_vars += list(model_epsilon.parameters())
        elif args.epsilon_network == 'params' or args.epsilon_network == 'proj_param':
            if args.regularizer == 'mrtv':
                # Gs_einc = torch.einsum('bik,bkj->bijk', self.field_info['Rec_mat'], self.field_info['E_inc']).reshape(self.num_freqs,-1,self.field_info['Rec_mat'].shape[-1])
                # init_chi = (torch.linalg.inv(torch.conj(Gs_einc.transpose(1,2))@Gs_einc+1e-5*torch.eye(self.field_info['Rec_mat'].shape[-1]))
                #             @torch.conj(Gs_einc.transpose(1,2)))@self.E_sca.reshape(self.num_freqs,-1).t()
                # init_chi = init_chi.squeeze().reshape(args.grid_num, args.grid_num).real
                from scipy.ndimage import gaussian_filter
                init_chi = np.load(args.params_path).astype(np.float32)
                init_chi = torch.from_numpy(gaussian_filter(init_chi, sigma=5.0))
            else:
                init_chi = torch.ones(args.grid_num, args.grid_num)
                # from scipy.ndimage import gaussian_filter
                # init_chi = np.load(args.params_path).astype(np.float32)
                # init_chi = torch.from_numpy(gaussian_filter(init_chi, sigma=2.0))
            epsilon = nn.Parameter(init_chi.to(self.device), requires_grad=True)
            # epsilon = nn.Parameter(torch.from_numpy(np.load('/root/shared-nvme/PINNs-IE/PINNs-ISP-IncompleteData/data/epsilon_australia.npy')).float().to(self.device), requires_grad=True)
            if args.epsilon_network == 'proj_param':
                model_epsilon = InversionDecoder(args.vae_model, args.vae_ckpt_path, latent_dim=args.vae_latent_dim, vae_config_path=args.vae_config_path, 
                                             max_params=args.max_params).to(self.device)
                model_epsilon.eval()
                for param in model_epsilon.parameters():
                    param.requires_grad = False
                init_latent_z0 = model_epsilon.get_latent(torch.ones(1, 1, 64, 64).to(self.device))
                init_latent_z1 = model_epsilon.get_latent(torch.from_numpy(np.load(args.params_path)).float().to(self.device).unsqueeze(0).unsqueeze(0))
                alpha = 0.0
                init_latent_z = alpha*init_latent_z1 + (1-alpha)*init_latent_z0
                # init_latent_z = torch.randn((1, args.vae_latent_dim)).to(self.device)
                latent_code = nn.Parameter(init_latent_z, requires_grad=True)
                projector_type = getattr(args, 'projector_type', 'vae')
                if projector_type == 'tv':
                    self.projector = TVProjector(args)
                else:
                    self.projector = VAEProjector(model_epsilon, latent_code, args)
        elif args.epsilon_network == 'paramzation':
            model_epsilon = InversionDecoder(args.vae_model, args.vae_ckpt_path, latent_dim=args.vae_latent_dim, vae_config_path=args.vae_config_path, 
                                             max_params=args.max_params).to(self.device)
            model_epsilon.eval()
            for param in model_epsilon.parameters():
                param.requires_grad = False
            init_latent_z0 = model_epsilon.get_latent(torch.ones(1, 1, 64, 64).to(self.device))
            init_latent_z1 = model_epsilon.get_latent(torch.from_numpy(np.load(args.params_path)).float().to(self.device).unsqueeze(0).unsqueeze(0))
            alpha = 0.0
            init_latent_z = alpha*init_latent_z1 + (1-alpha)*init_latent_z0
            # init_latent_z = torch.randn((1, args.vae_latent_dim)).to(self.device)
            epsilon = nn.Parameter(init_latent_z, requires_grad=True)
        elif args.epsilon_network == 'ft_paramz':
            model_epsilon = InversionDecoder(args.vae_model, args.vae_ckpt_path, latent_dim=args.vae_latent_dim, vae_config_path=args.vae_config_path, 
                                             max_params=args.max_params).to(self.device)
            init_latent_z0 = model_epsilon.get_latent(torch.ones(1, 1, 64, 64).to(self.device))
            init_latent_z1 = model_epsilon.get_latent(torch.from_numpy(np.load(args.params_path)).float().to(self.device).unsqueeze(0).unsqueeze(0))
            alpha = 0.0
            init_latent_z = alpha*init_latent_z1 + (1-alpha)*init_latent_z0
            # init_latent_z = torch.randn((1, args.vae_latent_dim)).to(self.device)
            epsilon = nn.Parameter(init_latent_z, requires_grad=True)
        elif args.epsilon_network == 'paramzations':
            init_chi = torch.ones(args.grid_num, args.grid_num)
            params_epsilon = nn.Parameter(init_chi.to(self.device))
            model_epsilon = InversionDecoder(args.vae_model, args.vae_ckpt_path, latent_dim=args.vae_latent_dim, vae_config_path=args.vae_config_path, 
                                             max_params=args.max_params, params_epsilon=params_epsilon).to(self.device)
            model_epsilon.eval()
            for param in model_epsilon.parameters():
                param.requires_grad = False
            model_epsilon.params_epsilon.requires_grad = True
            init_latent_z0 = model_epsilon.get_latent(torch.ones(1, 1, 64, 64).to(self.device))
            init_latent_z1 = model_epsilon.get_latent(torch.from_numpy(np.load(args.params_path)).float().to(self.device).unsqueeze(0).unsqueeze(0))
            alpha = 0.0
            init_latent_z = alpha*init_latent_z1 + (1-alpha)*init_latent_z0
            # init_latent_z = torch.randn((1, args.vae_latent_dim)).to(self.device)
            epsilon = nn.Parameter(init_latent_z, requires_grad=True)
        if len(args.freq)>1 and args.J_network == 'multi-mlp':
            optimizer_model = []
            # for i in range(len(args.freq)):
            #     optimizer_model.append(torch.optim.Adam(params=list(grad_vars[i]), lr=args.lrate, betas=(0.9, 0.999)))
            grad_vars_tot = []
            for i in range(len(args.freq)):
                grad_vars_tot+=(list(grad_vars[i]))
            optimizer_model.append(torch.optim.Adam(params=grad_vars_tot, lr=args.lrate, betas=(0.9, 0.999)))
        else:
            optimizer_model = [torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))]
        if 'isp' in args.method:
            if args.epsilon_network == 'params' or args.epsilon_network == 'paramzation' or args.epsilon_network == 'proj_param':
                optimizer_params = torch.optim.Adam(params=[epsilon], lr=args.params_lrate, betas=(0.9, 0.999))
            elif args.epsilon_network == 'ft_paramz':
                optimizer_params = torch.optim.Adam([{'params': [epsilon], 'lr': args.params_lrate, 'betas': (0.9, 0.999)},
                                                     {'params': model_epsilon.parameters(), 'lr': args.vae_lrate, 'betas': (0.9, 0.999)}])
            elif args.epsilon_network == 'paramzations':
                optimizer_params = torch.optim.Adam(params=[epsilon, model_epsilon.params_epsilon], lr=args.params_lrate, betas=(0.9, 0.999))
        start = 0
        # Load checkpoints
        
        if args.ft_path is not None and args.ft_path != 'None':
            ckpts = [args.ft_path]
        elif args.reload_weight:
            ckpts = [os.path.join(args.exp_dir, f) for f in sorted(os.listdir(args.exp_dir)) if 'tar' in f]
        else:
            ckpts = []
        if len(ckpts) > 0 and not args.no_reload:
            print('Found ckpts', ckpts)
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            start = ckpt['global_step']
            optimizer_model.load_state_dict(ckpt['optimizer_model_state_dict'])
            if args.method == 'isp':
                if args.epsilon_network in ['params', 'paramzation', 'paramzations', 'ft_paramz', 'proj_param']:
                    optimizer_params.load_state_dict(ckpt['optimizer_params_state_dict'])
                elif args.epsilon_network == 'mlp':
                    model_epsilon.load_state_dict(ckpt['model_epsilon_state_dict'])
            model_J.load_state_dict(ckpt['model_J_state_dict'])
        render_kwargs_train = {'model_fwd_fn': model_fwd_fn, 'model_J': model_J}
        if args.epsilon_network == 'mlp':
            render_kwargs_train['model_epsilon'] = model_epsilon
        elif args.epsilon_network == 'params' or args.epsilon_network == 'proj_param':
            render_kwargs_train['params_epsilon'] = epsilon
        elif args.epsilon_network in ['paramzation', 'paramzations', 'ft_paramz']:
            render_kwargs_train['model_epsilon'] = model_epsilon
            render_kwargs_train['params_epsilon'] = epsilon
        optimizer = {'model': optimizer_model, 'params': optimizer_params if ('param' in args.epsilon_network) and 'isp' in args.method else None}
        regular2func = {'tv_l1': L_TV_L1, 'tv_l2': L_TV_L2, 'mdtv_l1': MultiDirectionalTV, 'mrtv': MultiDirectionalTV}
        self.regular_item = regular2func[args.regularizer]()
        return render_kwargs_train, render_kwargs_train, optimizer, start
    
    def green_asembly(self, args):
        L_doi = args.L_doi
        grid_num = args.grid_num
        N_inc = args.N_inc
        N_rec = args.N_rec
        args.N_cell = grid_num * grid_num
        N_cell = args.N_cell
        
        y_dom,x_dom = torch.meshgrid([torch.arange(L_doi/2,-L_doi/2-L_doi/(grid_num-1)/2,-L_doi/(grid_num-1)),
                                      torch.arange(-L_doi/2,L_doi/2+L_doi/(grid_num-1)/2,L_doi/(grid_num-1))])

        def sample_angles(n, angle_range, mask_offset=0.0):
            """Uniform angles within [start_deg, end_deg), optionally rotated by
            `mask_offset` (fraction of one inter-angle step) to enable
            multi-mask statistics under fixed Tx/Rx counts."""
            start_deg, end_deg = angle_range
            start = torch.tensor(start_deg, device=self.device) * torch.pi / 180
            end = torch.tensor(end_deg, device=self.device) * torch.pi / 180
            if end <= start:
                end = start + 2 * torch.pi
            base = torch.linspace(start, end, steps=n + 1, device=self.device)[:-1]
            if mask_offset != 0.0:
                step = (end - start) / max(n, 1)
                base = base + step * mask_offset
            return base

        # mask_seed (R2-Comment 2 multi-mask): rotate Tx/Rx angles by a
        # fraction of one inter-angle step so that different mask_seeds give
        # statistically independent measurement geometries with the same N.
        ms = getattr(args, 'mask_seed', None)
        if ms is None:
            tx_off = rx_off = 0.0
        else:
            rng = np.random.default_rng(int(ms))
            tx_off = float(rng.random())
            rx_off = float(rng.random())
        theta_t = sample_angles(N_inc, args.tx_angle_range, tx_off)
        theta_r = sample_angles(N_rec, args.rx_angle_range, rx_off)
        xy_t = torch.cat([torch.cos(theta_t).unsqueeze(-1)*args.R_t, torch.sin(theta_t).unsqueeze(-1)*args.R_t], -1)
        xy_r = torch.cat([torch.cos(theta_r).unsqueeze(-1)*args.R_r, torch.sin(theta_r).unsqueeze(-1)*args.R_r], -1)
        if args.double_rec:
            gap = self.lam_0
            xy_r_new = torch.cat([torch.cos(theta_r).unsqueeze(-1)*(args.R_r+gap), torch.sin(theta_r).unsqueeze(-1)*(args.R_r+gap)], -1)
            xy_r = torch.cat((xy_r, xy_r_new), 0)
            N_rec *= 2

        x_dom = x_dom.to(self.device)
        y_dom = y_dom.to(self.device)
        xy_dom = torch.stack([x_dom, y_dom], -1)
        xy_t = xy_t.to(self.device)
        xy_r = xy_r.to(self.device)
        self.coords_eps = xy_dom
        self.coords_inc = torch.cat((torch.reshape(xy_dom.transpose(0, 1), [-1, 2]).unsqueeze(-2).repeat([1, N_inc, 1]),
                                    xy_t.unsqueeze(0).repeat([N_cell, 1, 1])), -1)
        if len(self.args.freq)>1 and self.args.J_network == 'single-mlp':
            self.coords_inc = torch.cat((self.k_0.view(-1,1,1,1).repeat([1, N_cell, N_inc, 1]),self.coords_inc[None,...].repeat([self.num_freqs, 1,1,1])), -1)
        else:
            self.coords_inc = self.coords_inc[None,...]
        # Gd --> Phi_mat, Green fuunction in DOI
        y_dom_flatten = y_dom.T.reshape([-1,1])
        x_dom_flatten = x_dom.T.reshape([-1,1])
        dist_cell = torch.sqrt((x_dom_flatten.repeat([1,N_cell])-x_dom_flatten.repeat([1,N_cell]).T)**2+
                        (y_dom_flatten.repeat([1,N_cell])-y_dom_flatten.repeat([1,N_cell]).T)**2)
        dist_cell = dist_cell + torch.eye(N_cell).to(self.device)
        # Phi_mat =  1j*self.k_0.view(-1,1,1)*self.eta_0 *(1j/4)*besselh(0,1,self.k_0.view(-1,1,1)*dist_cell[None,...])
        # Phi_mat = Phi_mat*(torch.ones(N_cell)-torch.eye(N_cell))[None,...].to(self.device)
        # Phi_mat = Phi_mat+(((1j*self.k_0*self.eta_0 *1j/4)*(2/(self.k_0*self.a_eqv)*besselh(1,1,self.k_0*self.a_eqv)+4*1j/(self.k_0**2*self.cell_area))).view(-1,1,1)*torch.eye(N_cell)[None,...]).to(self.device) 
        kba = self.k_0*self.a_eqv
        coeff = -1j*torch.pi*kba/2
        Phi_mat = (coeff * sp.bessel_j1(kba)).view(-1,1,1) * besselh(0,2,self.k_0.view(-1,1,1)*dist_cell[None,...])
        Phi_mat = Phi_mat*(torch.ones(N_cell)-torch.eye(N_cell))[None,...].to(self.device)
        Phi_mat = Phi_mat+((coeff*(besselh(1,2,self.k_0*self.a_eqv))-1).view(-1,1,1)*torch.eye(N_cell)[None,...]).to(self.device)
        
        # Gs --> R_mat, Green function in Receiver Domain
        rho_mat_r = torch.sqrt((xy_r[:,0].unsqueeze(-1).repeat([1,N_cell])-x_dom_flatten.repeat([1,N_rec]).T)**2 
                        +(xy_r[:,1].unsqueeze(-1).repeat([1,N_cell])-y_dom_flatten.repeat([1,N_rec]).T)**2)
        # R_mat = 1j*self.k_0.view(-1,1,1)*self.eta_0 *(1j/4)*besselh(0,1,self.k_0.view(-1,1,1)*rho_mat_r[None,...])
        R_mat = (coeff * sp.bessel_j1(kba)).view(-1,1,1) * besselh( 0, 2, self.k_0.view(-1,1,1)*rho_mat_r[None,...])
        
        if args.inc_wave == 'cir':
            # Gt --> T_mat, Green function in Transmitter Domain
            rho_mat_t = torch.sqrt((xy_t[:,0].unsqueeze(-1).repeat([1,N_cell])-x_dom_flatten.repeat([1,N_inc]).T)**2 
                            +(xy_t[:,1].unsqueeze(-1).repeat([1,N_cell])-y_dom_flatten.repeat([1,N_inc]).T)**2)
            # T_mat = 1j*self.k_0.view(-1,1,1)*self.eta_0 *(1j/4)*besselh(0,1,self.k_0.view(-1,1,1)*rho_mat_t[None,...])
            T_mat = 1/(4*1j)*besselh(0,2,self.k_0.view(-1,1,1)*rho_mat_t[None,...])
            E_inc = T_mat.transpose(1, 2)
            rho_mat_t_r = torch.sqrt((xy_r[:,0].unsqueeze(-1).repeat([1,N_inc])-xy_t[:,0].unsqueeze(-1).repeat([1,N_rec]).T)**2
                                     +(xy_r[:,1].unsqueeze(-1).repeat([1,N_inc])-xy_t[:,1].unsqueeze(-1).repeat([1,N_rec]).T)**2)
            # E_inc_tr = 1j*self.k_0.view(-1,1,1)*self.eta_0 *(1j/4)*besselh(0,1,self.k_0.view(-1,1,1)*rho_mat_t_r[None,...])
            E_inc_tr = 1/(4*1j)*besselh(0,2,self.k_0.view(-1,1,1)*rho_mat_t_r[None,...])
        elif args.inc_wave == 'plane':
            k_x = self.k_0.view(-1,1)*torch.cos(theta_t)[None,...]
            k_y = self.k_0.view(-1,1)*torch.sin(theta_t)[None,...]
            E_inc = torch.exp(x_dom_flatten[None,...]*k_x.unsqueeze(1)*1j +y_dom_flatten[None,...]*k_y.unsqueeze(1)*1j)
            E_inc_tr = torch.exp(xy_r[:,0:1].unsqueeze(0)*k_x.unsqueeze(1)*1j +xy_r[:,1:2].unsqueeze(0)*k_y.unsqueeze(1)*1j)
        self.field_info = {'E_inc': E_inc, 'Phi_mat': Phi_mat, 'Rec_mat': R_mat, 'E_inc_tr': E_inc_tr}
        if self.args.rm_same_points and self.args.R_t==self.args.R_r:
            trans_n = self.args.N_inc
            freq_n = self.num_freqs
            receiv_n = int(self.args.N_rec*2/3)+1
            masks_full = fresnel_data_preprocess(freq_n, receiv_n, trans_n, 360, self.args.grid_num**2, self.device)
            self.field_info['Rec_mat'] = masks_full[None, ...].repeat(freq_n,1,1,1) @ self.field_info['Rec_mat'][:,None,...].repeat(1,trans_n,1,1)
            self.field_info['E_inc_tr'] = (masks_full[None,...].repeat(freq_n,1,1,1)@torch.nan_to_num(self.field_info['E_inc_tr']).permute(0,2,1).unsqueeze(-1)).squeeze(-1).permute(0,2,1)    
    
    def render(self, xi_E_inc, xi_forward_mat, Phi_mat, Rec_mat, epsilon, input_J, **render_kwargs_train):
        re = {}
        if 'isp' in self.args.method:
            if self.args.epsilon_network == 'params' or self.args.epsilon_network == 'proj_param':
                epsilon = render_kwargs_train['params_epsilon']
                epsilon = epsilon_control(epsilon, self.args.params_constraint)
            elif self.args.epsilon_network == 'mlp':
                epsilon = render_kwargs_train['model_fwd_fn'](self.coords_eps, render_kwargs_train['model_epsilon']).squeeze(-1)
            elif self.args.epsilon_network in ['paramzation', 'paramzations', 'ft_paramz']:
                epsilon = render_kwargs_train['model_epsilon'](render_kwargs_train['params_epsilon'])
            # xi_all = -1j * self.omega.view(-1,1,1) * ((epsilon - 1) * self.eps_0 * self.cell_area)[None,...]
            xi_all = ((epsilon - 1)[None,...]*(1+0j)).repeat(self.num_freqs,1,1)
            xi_forward = torch.reshape(xi_all.transpose(1,2), [self.num_freqs, -1, 1])
            xi_forward_mat = torch.diag_embed(xi_forward.squeeze(-1))
            xi_E_inc = xi_forward_mat @ self.field_info['E_inc']
            norm_xi_E_inc = torch.mean(xi_E_inc.real ** 2 + xi_E_inc.imag ** 2)
            re['norm_xi_E_inc'] = norm_xi_E_inc
        re['epsilon'] = epsilon
        J = render_kwargs_train['model_fwd_fn'](input_J, render_kwargs_train['model_J'])
        J = torch.complex(J[..., 0], J[..., 1])
        re['J'] = J
        J_ = J.detach()
        re['J_nograd'] = J_
        re['Esca'] = Rec_mat @ J if len(Rec_mat.shape) == 3 else ((Rec_mat @ J.transpose(1,2).unsqueeze(-1)).squeeze(-1)).transpose(1,2)
        re['J_state'] = xi_E_inc + xi_forward_mat @ Phi_mat @ J
        return re
    
    def read_calibrate_fresnel(self, args):
        # 使用所选频率 (MATLAB中的use_fre_No)
        use_fre_no = np.array([int(i-2) for i in args.freq if i != ''])
        # use_fre_no = [1,2,4,5]
        freqs_all = torch.tensor([2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9, 10e9], device=self.device)
        freqs = freqs_all[torch.tensor(use_fre_no)]  # 索引从0开始，所以减1
        freq_n = len(freqs)
        raw_receiv_n = getattr(args, 'N_rec_data', 241)
        raw_trans_n = args.N_inc if getattr(args, 'N_inc_data', None) is None else args.N_inc_data
        p_sca_truth1, p_sca_truth_raw = read_exp_data(args.recdata_path, 9, raw_trans_n, raw_receiv_n)
        p_sca_truth1 = p_sca_truth1.to(self.device)
        # 原始数据有9个频率
        # 只使用选定的频率
        p_sca_truth_selected = torch.zeros((freq_n, raw_trans_n, raw_receiv_n), dtype=torch.complex64, device=self.device)
        for ii in range(freq_n):
            p_sca_truth_selected[ii,...] = p_sca_truth1[use_fre_no[ii],...]
        # p_sca_truth1 = torch.zeros_like(p_sca_truth_selected, device=self.device)
        # for i in range(freq_n):
        #     p_sca_truth1[i,...] = torch.from_numpy(wden_pytorch(p_sca_truth_selected[i,...].real.cpu().numpy(), level=4)+1j*wden_pytorch(p_sca_truth_selected[i,...].imag.cpu().numpy(), level=4)).to(self.device)
        # 子采样收发机（均匀选取）
        target_trans = args.N_inc_use if getattr(args, 'N_inc_use', None) is not None else raw_trans_n
        target_rec = args.N_rec_use if getattr(args, 'N_rec_use', None) is not None else raw_receiv_n
        target_trans = int(min(target_trans, raw_trans_n))
        target_rec = int(min(target_rec, raw_receiv_n))
        if target_trans < 1 or target_rec < 1:
            raise ValueError("N_inc_use 和 N_rec_use 需要大于等于 1")
        tx_idx = torch.linspace(0, raw_trans_n-1, steps=target_trans, device=self.device).round().long()
        rx_idx = torch.linspace(0, raw_receiv_n-1, steps=target_rec, device=self.device).round().long()
        p_sca_truth_selected = p_sca_truth_selected[:, tx_idx][:, :, rx_idx]

        # 同步裁剪各类场信息/坐标
        self.field_info['E_inc'] = self.field_info['E_inc'][:, :, tx_idx]
        self.field_info['E_inc_tr'] = self.field_info['E_inc_tr'][:, :, tx_idx]
        self.coords_inc = self.coords_inc[:, :, tx_idx, :]

        # 构建掩码，将 360 接收映射到子采样后的接收
        masks_full = fresnel_data_preprocess(freq_n, raw_receiv_n, raw_trans_n, 360, args.grid_num**2, self.device)
        masks_full = masks_full[tx_idx][:, rx_idx, :]
        self.field_info['Rec_mat'] = masks_full[None, ...].repeat(freq_n,1,1,1) @ self.field_info['Rec_mat'][:,None,...].repeat(1,target_trans,1,1)
        self.field_info['E_inc_tr'] = (masks_full[None,...].repeat(freq_n,1,1,1)@torch.nan_to_num(self.field_info['E_inc_tr']).permute(0,2,1).unsqueeze(-1)).squeeze(-1).permute(0,2,1)
        return p_sca_truth_selected
    
    def vie_fwd(self, args):
        epsilon_gt = np.load(args.params_path)
        epsilon_gt = torch.Tensor(epsilon_gt).to(self.device)
        epsilon = epsilon_gt.clone()
        # xi_all = -1j * self.omega.view(-1,1,1) * ((epsilon - 1) * self.eps_0 * self.cell_area)[None,...]
        xi_all = (((epsilon - 1)[None,...])*(1+0j)).repeat(self.num_freqs,1,1)
        xi_forward = torch.reshape(xi_all.transpose(1,2), [self.num_freqs, -1, 1])
        xi_forward_mat = torch.diag_embed(xi_forward.squeeze(-1))
        xi_E_inc = xi_forward_mat @ self.field_info['E_inc']
        norm_xi_E_inc = torch.mean(xi_E_inc.real ** 2 + xi_E_inc.imag ** 2)
        J_trad = torch.linalg.solve((torch.eye(args.N_cell)[None,...]-xi_forward_mat @ self.field_info['Phi_mat']),xi_E_inc)
        E_sca = self.field_info['Rec_mat'] @ J_trad if len(self.field_info['Rec_mat'].shape) == 3 else ((self.field_info['Rec_mat'] @ J_trad.transpose(1,2).unsqueeze(-1)).squeeze(-1)).transpose(1,2)
        return J_trad, E_sca    
        
    def train(self, args):
        epsilon_gt = np.load(args.params_path)
        epsilon_gt = torch.Tensor(epsilon_gt).to(self.device)
        epsilon = epsilon_gt.clone()
        if args.grid_num != epsilon.shape[0]:
            epsilon_gt = F.interpolate(epsilon_gt.unsqueeze(0).unsqueeze(0), size=(args.grid_num, args.grid_num), mode='bicubic', align_corners=False).squeeze()
            epsilon = F.interpolate(epsilon.unsqueeze(0).unsqueeze(0), size=(args.grid_num, args.grid_num), mode='bicubic', align_corners=False).squeeze()
        if args.method == 'fwd':
            loss_list = ['J_state_loss','Esca_loss','Jtrue_mse','BP_loss']
        elif 'isp' in args.method:
            if args.recdata_path.endswith('.npy'):
                ds_solution = np.load(args.recdata_path, allow_pickle=True).item()['E_sca']
            else:
                ds_solution = self.read_calibrate_fresnel(args).permute(0,2,1)
            E_sca = torch.tensor(ds_solution).to(self.device)
            # RNG-aligned noise: if --load_noisy_data is given, override E_sca with the saved tensor;
            # otherwise generate noise as usual, and optionally save it for downstream methods.
            if getattr(args, 'load_noisy_data', None):
                # Consume the same amount of RNG that torch.randn_like / torch.rand_like would have
                # used in the noise-generation branch, so the downstream model_J init RNG state
                # matches the run that originally generated the noise.
                _ = torch.randn_like(E_sca)
                loaded = np.load(args.load_noisy_data)
                E_sca = torch.tensor(loaded).to(self.device).to(E_sca.dtype)
                print(f"[RNG-ALIGN] loaded noisy E_sca from {args.load_noisy_data}, shape={tuple(E_sca.shape)}")
            elif args.noise_ratio > 0:
                if args.noise_type == 'gaussion':
                    noise = torch.randn_like(E_sca) * args.noise_ratio * torch.mean(torch.abs(E_sca))/np.sqrt(2)
                elif args.noise_type == 'uniform':
                    noise = E_sca * args.noise_ratio * torch.complex(2 * torch.rand_like(E_sca.real) - 1, 2 * torch.rand_like(E_sca.real) - 1)/torch.sqrt(torch.tensor(2.0/3.0))
                elif args.noise_type == 'uniform_try':
                    n = 2 * torch.rand_like(E_sca) - (1 + 1j)
                    noise = n * args.noise_ratio * torch.mean(torch.abs(E_sca))/np.sqrt(2/3)
                elif args.noise_type == 'gaussion_new':
                    n = torch.randn_like(E_sca)
                    noise = n * (torch.norm(E_sca) / torch.norm(n)) * (args.noise_ratio if args.noise_ratio <= 1.0 else 10**(-args.noise_ratio/20.0))
                elif args.noise_type == 'uniform_new':
                    n = 2 * torch.rand_like(E_sca) - (1 + 1j)
                    noise = n * (torch.norm(E_sca) / torch.norm(n)) * (args.noise_ratio if args.noise_ratio <= 1.0 else 10**(-args.noise_ratio/20.0))
                E_sca = E_sca + noise
                if getattr(args, 'save_noisy_data', None):
                    import os as _os
                    _os.makedirs(_os.path.dirname(args.save_noisy_data) or '.', exist_ok=True)
                    np.save(args.save_noisy_data, E_sca.detach().cpu().numpy())
                    print(f"[RNG-ALIGN] saved noisy E_sca to {args.save_noisy_data}, shape={tuple(E_sca.shape)}")
            loss_list = ['J_state_loss','Esca_loss','TV_loss','BP_loss']
            if args.save_metric:
                loss_list += ['MSE', 'SSIM']
            if getattr(args, 'log_diagnostics', False):
                loss_list += ['chi_norm']
            self.E_sca = E_sca
            
        # xi_all = -1j * self.omega.view(-1,1,1) * ((epsilon - 1) * self.eps_0 * self.cell_area)[None,...]
        xi_all = (((epsilon - 1)[None,...])*(1+0j)).repeat(self.num_freqs,1,1)
        xi_forward = torch.reshape(xi_all.transpose(1,2), [self.num_freqs, -1, 1])
        xi_forward_mat = torch.diag_embed(xi_forward.squeeze(-1))
        xi_E_inc = xi_forward_mat @ self.field_info['E_inc']
        norm_xi_E_inc = torch.mean(xi_E_inc.real ** 2 + xi_E_inc.imag ** 2)
        
        if args.method == 'fwd':
            # Direct solution
            ds_fwd_savedir = os.path.join(args.exp_dir, 'ds_solution.npy')
            save_plot_path = os.path.join(args.exp_dir, 'ds_solution.png')
            J_trad = torch.linalg.solve((torch.eye(args.N_cell)[None,...]-xi_forward_mat @ self.field_info['Phi_mat']),xi_E_inc)
            E_sca = self.field_info['Rec_mat'] @ J_trad if len(self.field_info['Rec_mat'].shape) == 3 else ((self.field_info['Rec_mat'] @ J_trad.transpose(1,2).unsqueeze(-1)).squeeze(-1)).transpose(1,2)
            ds_solution = {'J_trad': J_trad, 'E_sca': E_sca}
            np.save(ds_fwd_savedir, ds_solution)
            plot_J_figure_multifreqs(J_trad.squeeze(-1).cpu().numpy(), save_plot_path, args)

        # Create nerf model
        render_kwargs_train, render_kwargs_test, optimizer, start = self.create_isp_neuropretor(args)
        global_step = start
        if args.render_only:
            self.test(args, 'renderonly_{}_{:06d}.npy'.format('test' if args.render_test else 'path', start), render_kwargs_test, 'renderonly_{}_{:06d}.png'.format('test' if args.render_test else 'path', start), epsilon)
        if 'isp' in self.args.method:
            if self.args.epsilon_network == 'params' or self.args.epsilon_network == 'proj_param':
                epsilon = render_kwargs_train['params_epsilon']
                epsilon = epsilon_control(epsilon, args.params_constraint)
            elif self.args.epsilon_network == 'mlp':
                epsilon = render_kwargs_train['model_fwd_fn'](self.coords_eps, render_kwargs_train['model_epsilon']).squeeze(-1)
            elif self.args.epsilon_network in ['paramzation', 'paramzations', 'ft_paramz']:
                epsilon = render_kwargs_train['model_epsilon'](render_kwargs_train['params_epsilon'])
        self.test(args, 'testset_000000.npy', render_kwargs_test, None, epsilon)    # 'testset_000000.png'

        N_iters = args.max_iter + 1
        loss_recorder = LossRecorder(args.exp_dir, loss_names=loss_list)
        print('Begin training...')
        start = start + 1
        last_regular_loss = 0
        last_fd = 1e-6*self.cell_area**2
        solve_time = 0
        data_ma, reg_ma = None, None
        for i in trange(start, N_iters):
            epoch_start_time = time()
            re = self.render(xi_E_inc, xi_forward_mat, self.field_info['Phi_mat'], self.field_info['Rec_mat'], epsilon, input_J=self.coords_inc, **render_kwargs_train)
            if isinstance(optimizer['model'], list):
                for kk in optimizer['model']:
                    kk.zero_grad()
            else:
                optimizer['model'].zero_grad()
            if 'isp' in args.method and 'param' in args.epsilon_network:
                optimizer['params'].zero_grad()
            # E_sca_evl_loss = (img2mse(re['Esca'].real, E_sca.real) + img2mse(re['Esca'].imag, E_sca.imag))/torch.mean(E_sca.real **2 + E_sca.imag **2)
            # J_state_loss = (img2mse(re['J_state'].real, re['J'].real) + img2mse(re['J_state'].imag, re['J'].imag))/torch.clamp(norm_xi_E_inc, min=1e-6)#/(norm_xi_E_inc + torch.finfo(torch.float32).eps)
            if args.method == 'fwd':
                J_state_loss = (img2mse(re['J_state'].real, re['J'].real) + img2mse(re['J_state'].imag, re['J'].imag))/norm_xi_E_inc
                E_sca_evl_loss = (img2mse(re['Esca'].real, E_sca.real) + img2mse(re['Esca'].imag, E_sca.imag))/torch.mean(E_sca.real **2 + E_sca.imag **2)
                J_evl_loss = (img2mse(re['J'].real, J_trad.real) + img2mse(re['J'].imag, J_trad.imag))/torch.mean(J_trad.real **2+J_trad.imag **2)
                loss = J_state_loss
                current_losses = {'J_state_loss': J_state_loss.item(),'Esca_loss': E_sca_evl_loss.item(),
                                    'Jtrue_mse': J_evl_loss.item(), 'BP_loss': loss.item()}
                loss_recorder.update(current_losses)
            elif 'isp' in args.method:
                regular_loss = self.regular_item(re['epsilon'])
                if 'fdtot' in args.method:
                    E_sca_evl_loss = (img2mse((re['Esca']+self.field_info['E_inc_tr']).real, (E_sca+self.field_info['E_inc_tr']).real) + img2mse((re['Esca']+self.field_info['E_inc_tr']).imag, (E_sca+self.field_info['E_inc_tr']).imag))/torch.mean((E_sca).real **2 + (E_sca).imag **2)
                elif 'fd' in args.method:
                    E_sca_evl_loss = (img2mse(re['Esca'].real, E_sca.real) + img2mse(re['Esca'].imag, E_sca.imag))/torch.mean(E_sca.real **2 + E_sca.imag **2)
                elif 'pdtot' in args.method:
                    E_sca_evl_loss = 2*(img2mse(torch.abs(re['Esca']+self.field_info['E_inc_tr']), torch.abs(E_sca+self.field_info['E_inc_tr'])))/torch.mean(torch.abs(E_sca)**2)#torch.mean(torch.abs(E_sca+self.field_info['E_inc_tr'])**2-torch.abs(self.field_info['E_inc_tr'])**2)
                elif 'pd' in args.method:
                    E_sca_evl_loss = (img2mse(torch.abs(re['Esca']), torch.abs(E_sca)))/torch.mean(torch.abs(E_sca)**2)
                elif 'ad' in args.method:
                    E_sca_evl_loss = (img2mse(torch.angle(re['Esca']), torch.angle(E_sca)))/torch.mean(torch.angle(E_sca)**2)
                if 'param' in args.epsilon_network:
                    J_state_loss = (img2mse(re['J_state'].real, re['J'].real) + img2mse(re['J_state'].imag, re['J'].imag))/(re['norm_xi_E_inc'] + torch.finfo(torch.float32).eps)
                    if args.regularizer == 'mrtv':
                        if i==0:
                            last_regular_loss = regular_loss.item()
                            last_fd = J_state_loss.item()*self.cell_area**2
                        loss = (E_sca_evl_loss + J_state_loss)*((regular_loss+last_fd)/(last_regular_loss+last_fd))
                        if i % args.i_regularizer == 0:
                            last_regular_loss = regular_loss.item()
                            last_fd = J_state_loss.item()*self.cell_area**2
                    elif args.regularizer == 'tv_l1' or args.regularizer == 'tv_l2' or args.regularizer == 'mdtv_l1':
                        loss = E_sca_evl_loss + J_state_loss + args.regularizer_weight*regular_loss
                    else:
                        loss = E_sca_evl_loss + J_state_loss
                    # if args.epsilon_network == 'paramzation':
                    #     latent_kl = kl_divergence(render_kwargs_train['params_epsilon'])
                    #     loss += 5e-3 * latent_kl
                elif args.epsilon_network == 'mlp':
                    J_state_loss = (img2mse(re['J_state'].real, re['J'].real) + img2mse(re['J_state'].imag, re['J'].imag))/(re['norm_xi_E_inc'] + torch.finfo(torch.float64).eps)
                    if i<1000:
                        loss = E_sca_evl_loss + J_state_loss
                    else:
                        loss = E_sca_evl_loss + J_state_loss + 0.01*regular_loss
                current_losses = {'J_state_loss': J_state_loss.item(),'Esca_loss': E_sca_evl_loss.item(),
                                    'TV_loss': regular_loss.item(), 'BP_loss': loss.item()}
                if args.save_metric:
                    mse_gt = torch.sqrt(F.mse_loss(re['epsilon'].detach(), epsilon_gt).item()/torch.mean(epsilon_gt**2))
                    ssim_gt = self.ssim(re['epsilon'].detach().unsqueeze(0).unsqueeze(0),epsilon_gt.unsqueeze(0).unsqueeze(0))
                    current_losses.update({'MSE': mse_gt.item(), 'SSIM': ssim_gt.item()})
                if getattr(args, 'log_diagnostics', False):
                    # contrast magnitude norm (R2-2 convergence diagnostic)
                    chi_norm = torch.norm(re['epsilon'].detach() - 1.0).item()
                    current_losses['chi_norm'] = chi_norm
                loss_recorder.update(current_losses)

            loss.backward()
            if isinstance(optimizer['model'], list):
                for optim_tmp in optimizer['model']:
                    optim_tmp.step()
            else:
                optimizer['model'].step()
            if 'isp' in args.method and 'param' in args.epsilon_network:
                optimizer['params'].step()
            # if args.epsilon_network=='proj_param' and i%args.i_projection==0 and i!=((N_iters-1)//args.i_projection)*args.i_projection and i!=((N_iters-1)//args.i_projection-1)*args.i_projection:
            # Projection schedule: trigger when k > i_projection_start AND k % i_projection == 0.
            # i_projection_start defaults to 300, which (combined with i_projection=200)
            # gives projections at iterations {400, 600, 800, 1000} — the schedule reported
            # in Fig. 9 of the original submission. R2-Comment 3 ablation varies
            # i_projection_start over {0, 200, 400, 1000} to test projection timing.
            i_proj_start = getattr(args, 'i_projection_start', 300)
            i_proj_stop = getattr(args, 'i_projection_stop', N_iters)
            # Optional: linearly decay projection inner-loop TV weight over outer iters
            # (mirrors DeepCSI regularizer_decay schedule for the physics-step TV).
            _tv_final = getattr(args, 'proj_tv_weight_final', None)
            if _tv_final is not None:
                _tv_init = float(getattr(args, 'proj_tv_weight', 0.05))
                _first_proj = i_proj_start + args.i_projection
                _last_proj = N_iters
                if i >= _first_proj and _last_proj > _first_proj:
                    _frac = max(0.0, min(1.0, (i - _first_proj) / (_last_proj - _first_proj)))
                    _tv_cur = _tv_init + (float(_tv_final) - _tv_init) * _frac
                else:
                    _tv_cur = _tv_init
                if hasattr(self, 'projector') and self.projector is not None:
                    self.projector.tv_weight = _tv_cur
                if getattr(self, '_lazy_projector', None) is not None:
                    self._lazy_projector.tv_weight = _tv_cur
            if (args.epsilon_network=='proj_param'
                and not getattr(args, 'disable_projection', False)
                and args.i_projection>0
                and (i%args.i_projection==0 or i==N_iters-1)
                and i>i_proj_start
                and i<=i_proj_stop):
                self.projector.project(re['epsilon'])
            elif (args.epsilon_network=='params'
                  and getattr(args, 'enable_lazy_projection', False)
                  and args.i_projection>0
                  and (i%args.i_projection==0 or i==N_iters-1)
                  and i>i_proj_start
                  and i<=i_proj_stop):
                if getattr(self, '_lazy_projector', None) is None:
                    proj_type = getattr(args, 'projector_type', 'vae')
                    if proj_type == 'tv':
                        self._lazy_projector = TVProjector(args)
                    else:
                        _model_ep = InversionDecoder(args.vae_model, args.vae_ckpt_path,
                                                     latent_dim=args.vae_latent_dim,
                                                     vae_config_path=args.vae_config_path,
                                                     max_params=args.max_params).to(self.device)
                        _model_ep.eval()
                        for _p in _model_ep.parameters():
                            _p.requires_grad = False
                        _z_seed = _model_ep.get_latent(torch.ones(1, 1, 64, 64).to(self.device))
                        _latent_code = nn.Parameter(_z_seed.clone(), requires_grad=True)
                        self._lazy_projector = VAEProjector(_model_ep, _latent_code, args)
                self._lazy_projector.project(re['epsilon'])
            epoch_end_time = time()
            solve_time += epoch_end_time - epoch_start_time
            
            decay_begin = 500
            delta_regularizer_weight = (1-args.regularizer_decay)/((args.max_iter-decay_begin)//100)
            if global_step > decay_begin:
                args.regularizer_weight = args.regularizer_weight * (1-(global_step-decay_begin)//100*delta_regularizer_weight)
            # if 'isp' in args.method:
            #     if data_ma is None:
            #         data_ma = (E_sca_evl_loss + J_state_loss).detach()
            #         reg_ma = regular_loss.detach()
            #     if global_step >= decay_begin:
            #         with torch.no_grad():
            #             data_ma = 0.9 * data_ma + 0.1 * (E_sca_evl_loss + J_state_loss).detach()
            #             reg_ma = 0.9 * reg_ma + 0.1 * regular_loss.detach()
            #             if args.epsilon_network == 'params':
            #                 target_ratio = 0.18
            #                 min_w, max_w = 0.05, 0.2
            #                 adjust_low, adjust_high = 0.6, 1.25
            #             else:
            #                 # target_ratio = 0.3  # 0.15
            #                 # min_w, max_w = 0.1, 0.3
            #                 # adjust_low, adjust_high = 0.7, 1.3
            #                 target_ratio = 0.18
            #                 min_w, max_w = 0.05, 0.2
            #                 adjust_low, adjust_high = 0.6, 1.25
            #             cur_ratio = (args.regularizer_weight * reg_ma) / (data_ma + 1e-8)
            #             # adjust = torch.clamp((target_ratio / cur_ratio).sqrt(), 0.5, 2.0)
            #             adjust = torch.clamp((target_ratio / cur_ratio).sqrt(), adjust_low, adjust_high)
            #             new_weight = args.regularizer_weight * adjust
            #             # args.regularizer_weight = torch.clamp(new_weight, min=0.05, max=0.2).item()
            #             args.regularizer_weight = torch.clamp(new_weight, min=min_w, max=max_w).item()
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            # new_lrate = args.lrate * (0.5 ** (global_step / decay_steps))
            # new_lrate = new_lrate if new_lrate>1e-3 else 1e-3
            params_decay_steps = args.params_lrate_decay * 1000
            if i > 1000:
                args.params_lrate = 5e-2
            params_new_lrate = args.params_lrate * (decay_rate ** (global_step / params_decay_steps))
            vae_decay_steps = args.vae_lrate_decay * 1000
            vae_new_lrate = args.vae_lrate * (decay_rate ** (global_step / vae_decay_steps))
            new_lrate = {'model': new_lrate, 'params': params_new_lrate}
            for optimizer_name,optimizer_tmp in optimizer.items():
                if optimizer_tmp is not None:
                    if isinstance(optimizer_tmp, list):
                        for optim_tmp in optimizer_tmp:
                            for param_group in optim_tmp.param_groups:
                                param_group['lr'] = new_lrate[optimizer_name]
                    else:
                        if args.epsilon_network == 'ft_paramz':
                            for idx,param_group in enumerate(optimizer_tmp.param_groups):
                                if idx==0:
                                    param_group['lr'] = new_lrate[optimizer_name]
                                else:
                                    param_group['lr'] = vae_new_lrate
                        else:
                            for param_group in optimizer_tmp.param_groups:
                                param_group['lr'] = new_lrate[optimizer_name]

            if i % args.i_weights == 0:
                path = os.path.join(args.exp_dir, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'model_J_state_dict': [kk.state_dict() for kk in render_kwargs_train['model_J']],
                    'optimizer_model_state_dict': [kk.state_dict() for kk in optimizer['model']],
                    'model_epsilon_state_dict': render_kwargs_train['model_epsilon'].state_dict() if 'isp' in args.method and args.epsilon_network=='mlp' else None,
                    'optimizer_params_state_dict': optimizer['params'].state_dict() if 'isp' in args.method and 'param' in args.epsilon_network else None,
                }, path)
                print('Saved checkpoints at', path)

            if i % args.i_testset == 0 and i > 0:
                if i == args.max_iter:
                    self.test(args, 'testset_{:06d}.npy'.format(i), render_kwargs_test, 'testset_{:06d}.png'.format(i), re['epsilon'])
                else:
                    self.test(args, 'testset_{:06d}.npy'.format(i), render_kwargs_test, None, re['epsilon'])
                
            if i % args.i_print == 0:
                if args.method == 'fwd':
                    tqdm.write(f"[TRAIN] Iter: {i} J_state_loss: {J_state_loss.item()} E_sca_evl_loss: {E_sca_evl_loss.item()} J_evl_loss: {J_evl_loss.item()}")
                elif 'isp' in args.method:
                    tqdm.write(f"[TRAIN] Iter: {i} J_state_loss: {J_state_loss.item()} E_sca_evl_loss: {E_sca_evl_loss.item()} Regular_loss: {regular_loss.item()}")
            global_step += 1
        loss_recorder.plot_losses()
        loss_recorder.save_history()
        if 'isp' in args.method:
            mse_gt = torch.sqrt(F.mse_loss(re['epsilon'].detach(), epsilon_gt).item()/torch.mean(epsilon_gt**2))
            ssim_gt = self.ssim(re['epsilon'].detach().unsqueeze(0).unsqueeze(0),epsilon_gt.unsqueeze(0).unsqueeze(0))
            print('[TRAIN FINISH] model misfit:{}, data misfit:{}, ssim:{}, time cost:{}s'.format(mse_gt, np.sqrt(E_sca_evl_loss.item()), ssim_gt, solve_time))
        elif 'fwd' in args.method:
            print('[TRAIN FINISH] data misfit:{}, time cost:{}s'.format(np.sqrt(E_sca_evl_loss.item()), solve_time))
        
        if 'isp' in args.method and args.result_file is not None:
            exp_name = args.exp_dir.split('/')[-1]
            ninc = self.args.N_inc_use if self.args.N_inc_use is not None else self.args.N_inc
            nrec = self.args.N_rec_use if self.args.N_rec_use is not None else self.args.N_rec
            headers = ['Shape', 'Task', 'Freq', 'Method', 'NSR', 'TX/RX', 'Model Misfit', 'Data Misfit', 'SSIM', 'Time Cost']
            results = [exp_name.split('_')[1], exp_name.split('_')[2], exp_name.split('_')[7],
                       exp_name.split('_')[10], float(args.exp_dir.split('_')[-1][3:]), f'{ninc}/{nrec}', mse_gt.item(),
                       np.sqrt(E_sca_evl_loss.item()), ssim_gt.item(), solve_time]
            csv_writer(args.result_file, headers, results)
        
    def test(self, args, save_file, render_kwargs_test, save_plot_path=None, epsilon=None):
        testsavedir = os.path.join(args.exp_dir, save_file)
        with torch.no_grad():
            fn_test = render_kwargs_test['model_fwd_fn']
            output = fn_test(self.coords_inc, render_kwargs_test['model_J'])
            J_pred = output.squeeze(-1)
            Esca_pred = self.field_info['Rec_mat'] @ (J_pred[...,0]+1j*J_pred[...,1]) if len(self.field_info['Rec_mat'].shape) == 3 else ((self.field_info['Rec_mat'] @ (J_pred[...,0]+1j*J_pred[...,1]).transpose(1,2).unsqueeze(-1)).squeeze(-1)).transpose(1,2)
            ds_solution = {'J_pred': J_pred.cpu().numpy(), 'E_sca': Esca_pred.cpu().numpy()}
            if 'isp' in args.method and epsilon is not None:
                ds_solution['epsilon'] = epsilon.detach().cpu().numpy()
            np.save(testsavedir, ds_solution)
            if save_plot_path is not None:
                save_plot_path = os.path.join(args.exp_dir, save_plot_path)
                plot_J_figure_multifreqs(output.squeeze(-1).cpu().numpy(), save_plot_path, args)
                if 'isp' in args.method:
                    save_plot_path_params = save_plot_path[:-4] + '_params.png'
                    epsilon = epsilon.detach().cpu().numpy()
                    plot_params_figure(epsilon, save_plot_path_params)
        print('Saved test set')
