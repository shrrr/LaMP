import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from .utils import *
from .models import *
from .vae import *

class PINNsFwdIsp2D():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        self.eta_0 = 120 * np.pi
        c = 3e8
        self.eps_0 = 8.85e-12
        args.freq = float(args.freq)
        lam_0 = c / (args.freq * 1e9)
        self.k_0 = 2 * np.pi / lam_0
        self.omega = self.k_0 * c

        self.step_size = args.L_doi / (args.grid_num - 1)
        self.cell_area = self.step_size ** 2
        self.a_eqv = np.sqrt(self.cell_area/np.pi)
        self.green_asembly(args)

    def fwd_model(self, inputs, networkfwd_fn, embed_fn):
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = embed_fn(inputs_flat)
        outputs_flat = networkfwd_fn(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs
    
    def create_isp_neuropretor(self, args):
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        skips = [4]
        model_J = NeJF(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch * 2, output_ch=2, skips=skips, tanh=True).to(self.device)
        model_fwd_fn = lambda inputs, networkfwd_fn: self.fwd_model(inputs, networkfwd_fn, embed_fn=embed_fn)
        grad_vars = list(model_J.parameters())
        if args.epsilon_network == 'mlp':
            model_epsilon = NeJF(D=args.netdepth, W=args.netwidth,
                                input_ch=input_ch, output_ch=1, skips=skips, tanh=True, epsilon=True).to(self.device)
            grad_vars += list(model_epsilon.parameters())
        elif args.epsilon_network == 'params':
            epsilon = nn.Parameter(torch.ones(args.grid_num, args.grid_num).to(self.device), requires_grad=True)
        elif args.epsilon_network == 'paramzation':
            model_epsilon = InversionDecoder(args.vae_model, args.vae_ckpt_path, latent_dim=args.vae_latent_dim).to(self.device)
            model_epsilon.eval()
            for param in model_epsilon.parameters():
                param.requires_grad = False
            init_latent_z0 = model_epsilon.get_latent(torch.ones(1, 1, 64, 64).to(self.device))
            init_latent_z1 = model_epsilon.get_latent(torch.from_numpy(np.load('/root/shared-nvme/PINNs-IE/PINNs-ISP-IncompleteData/data/epsilon_twinCircle.npy')).float().to(self.device).unsqueeze(0).unsqueeze(0))
            alpha = 1.0
            init_latent_z = alpha*init_latent_z1 + (1-alpha)*init_latent_z0
            # init_latent_z = torch.randn((1, args.vae_latent_dim)).to(self.device)
            epsilon = nn.Parameter(init_latent_z, requires_grad=True)
        optimizer_model = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
        if 'isp' in args.method:
            if args.epsilon_network == 'params' or args.epsilon_network == 'paramzation':
                optimizer_params = torch.optim.Adam(params=[epsilon], lr=args.params_lrate, betas=(0.9, 0.999))

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
                if args.epsilon_network == 'params' or args.epsilon_network == 'paramzation':
                    optimizer_params.load_state_dict(ckpt['optimizer_params_state_dict'])
                elif args.epsilon_network == 'mlp':
                    model_epsilon.load_state_dict(ckpt['model_epsilon_state_dict'])
            model_J.load_state_dict(ckpt['model_J_state_dict'])
        render_kwargs_train = {'model_fwd_fn': model_fwd_fn, 'model_J': model_J}
        if args.epsilon_network == 'mlp':
            render_kwargs_train['model_epsilon'] = model_epsilon
        elif args.epsilon_network == 'params':
            render_kwargs_train['params_epsilon'] = epsilon
        elif args.epsilon_network == 'paramzation':
            render_kwargs_train['model_epsilon'] = model_epsilon
            render_kwargs_train['params_epsilon'] = epsilon
        optimizer = {'model': optimizer_model, 'params': optimizer_params if ('param' in args.epsilon_network) and 'isp' in args.method else None}
        self.regular_item = L_TV_L1(TVLoss_weight=1)
        return render_kwargs_train, render_kwargs_train, optimizer, start
    
    def green_asembly(self, args):
        L_doi = args.L_doi
        grid_num = args.grid_num
        N_inc = args.N_inc
        N_rec = args.N_rec
        args.N_cell = grid_num * grid_num
        N_cell = args.N_cell
        
        y_dom,x_dom = torch.meshgrid([torch.arange(L_doi/2,-L_doi/2-L_doi/(grid_num-1),-L_doi/(grid_num-1)),
                                      torch.arange(-L_doi/2,L_doi/2+L_doi/(grid_num-1),L_doi/(grid_num-1))])

        def sample_angles(n, angle_range):
            """Uniform angles within [start_deg, end_deg); wraps if end<=start."""
            start_deg, end_deg = angle_range
            start = torch.tensor(start_deg, device=self.device) * torch.pi / 180
            end = torch.tensor(end_deg, device=self.device) * torch.pi / 180
            if end <= start:
                end = start + 2 * torch.pi
            return torch.linspace(start, end, steps=n + 1, device=self.device)[:-1]

        theta_t = sample_angles(N_inc, args.tx_angle_range)
        theta_r = sample_angles(N_rec, args.rx_angle_range)
        xy_t = torch.cat([torch.cos(theta_t).unsqueeze(-1)*args.R_t, torch.sin(theta_t).unsqueeze(-1)*args.R_t], -1)
        xy_r = torch.cat([torch.cos(theta_r).unsqueeze(-1)*args.R_r, torch.sin(theta_r).unsqueeze(-1)*args.R_r], -1)

        x_dom = x_dom.to(self.device)
        y_dom = y_dom.to(self.device)
        xy_dom = torch.stack([x_dom, y_dom], -1)
        xy_t = xy_t.to(self.device)
        xy_r = xy_r.to(self.device)
        self.coords_eps = xy_dom
        self.coords_inc = torch.cat((torch.reshape(xy_dom.transpose(0, 1), [-1, 2]).unsqueeze(-2).repeat([1, N_inc, 1]),
                                xy_t.unsqueeze(0).repeat([N_cell, 1, 1])), -1)

        # Gd --> Phi_mat, Green fuunction in DOI
        y_dom_flatten = y_dom.T.reshape([-1,1])
        x_dom_flatten = x_dom.T.reshape([-1,1])
        dist_cell = torch.sqrt((x_dom_flatten.repeat([1,N_cell])-x_dom_flatten.repeat([1,N_cell]).T)**2+
                        (y_dom_flatten.repeat([1,N_cell])-y_dom_flatten.repeat([1,N_cell]).T)**2)
        dist_cell = dist_cell + torch.eye(N_cell).to(self.device)
        Phi_mat =  1j*self.k_0*self.eta_0 *(1j/4)*besselh(0,1,self.k_0*dist_cell)
        Phi_mat = Phi_mat*(torch.ones(N_cell)-torch.eye(N_cell)).to(self.device)
        Phi_mat = Phi_mat+(1j*self.k_0*self.eta_0 *(1j/4)*(2/(self.k_0*self.a_eqv)*besselh(1,1,self.k_0*self.a_eqv)+4*1j/(self.k_0**2*self.cell_area))*torch.eye(N_cell)).to(self.device)
        
        # Gs --> R_mat, Green function in Receiver Domain
        rho_mat_r = torch.sqrt((xy_r[:,0].unsqueeze(-1).repeat([1,N_cell])-x_dom_flatten.repeat([1,N_rec]).T)**2 
                        +(xy_r[:,1].unsqueeze(-1).repeat([1,N_cell])-y_dom_flatten.repeat([1,N_rec]).T)**2)
        R_mat = 1j*self.k_0*self.eta_0 *(1j/4)*besselh(0,1,self.k_0*rho_mat_r)
        
        if args.inc_wave == 'cir':
            # Gt --> T_mat, Green function in Transmitter Domain
            rho_mat_t = torch.sqrt((xy_t[:,0].unsqueeze(-1).repeat([1,N_cell])-x_dom_flatten.repeat([1,N_inc]).T)**2 
                            +(xy_t[:,1].unsqueeze(-1).repeat([1,N_cell])-y_dom_flatten.repeat([1,N_inc]).T)**2)
            T_mat = 1j*self.k_0*self.eta_0 *(1j/4)*besselh(0,1,self.k_0*rho_mat_t)
            E_inc = T_mat.T
        elif args.inc_wave == 'plane':
            k_x = self.k_0*torch.cos(theta_t)
            k_y = self.k_0*torch.sin(theta_t)
            E_inc = torch.exp(x_dom_flatten@k_x.unsqueeze(0)*1j +y_dom_flatten@k_y.unsqueeze(0)*1j)        
        self.field_info = {'E_inc': E_inc, 'Phi_mat': Phi_mat, 'Rec_mat': R_mat}
    
    def render(self, xi_E_inc, xi_forward_mat, Phi_mat, Rec_mat, epsilon, input_J, **render_kwargs_train):
        re = {}
        if 'isp' in self.args.method:
            if self.args.epsilon_network == 'params':
                epsilon = render_kwargs_train['params_epsilon']
            elif self.args.epsilon_network == 'mlp':
                epsilon = render_kwargs_train['model_fwd_fn'](self.coords_eps, render_kwargs_train['model_epsilon']).squeeze(-1)
            elif self.args.epsilon_network == 'paramzation':
                epsilon = render_kwargs_train['model_epsilon'](render_kwargs_train['params_epsilon'])
            xi_all = -1j * self.omega * (epsilon - 1) * self.eps_0 * self.cell_area
            xi_forward = torch.reshape(xi_all.t(), [-1, 1])
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
        re['Esca'] = Rec_mat @ J
        re['J_state'] = xi_E_inc + xi_forward_mat @ Phi_mat @ J
        return re
        
    def train(self, args):
        if args.method == 'fwd':
            epsilon = np.load(args.params_path)
            epsilon = torch.Tensor(epsilon).to(self.device)
            if args.grid_num != epsilon.shape[0]:
                epsilon = F.interpolate(epsilon.unsqueeze(0).unsqueeze(0), size=(args.grid_num, args.grid_num), mode='bicubic', align_corners=False).squeeze()
            loss_list = ['J_state_loss','Esca_loss','Jtrue_mse','BP_loss']
        elif 'isp' in args.method:
            ds_solution = np.load(args.recdata_path, allow_pickle=True).item()
            E_sca = torch.tensor(ds_solution['E_sca']).to(self.device)
            epsilon = torch.ones((args.grid_num, args.grid_num)).to(self.device)
            loss_list = ['J_state_loss','Esca_loss','TV_loss','BP_loss']
            
        xi_all = -1j * self.omega * (epsilon - 1) * self.eps_0 * self.cell_area
        xi_forward = torch.reshape(xi_all.t(), [-1, 1])
        xi_forward_mat = torch.diag_embed(xi_forward.squeeze(-1))
        xi_E_inc = xi_forward_mat @ self.field_info['E_inc']
        norm_xi_E_inc = torch.mean(xi_E_inc.real ** 2 + xi_E_inc.imag ** 2)
        
        if args.method == 'fwd':
            # Direct solution
            ds_fwd_savedir = os.path.join(args.exp_dir, 'ds_solution.npy')
            save_plot_path = os.path.join(args.exp_dir, 'ds_solution.png')
            J_trad = torch.linalg.solve((torch.eye(args.N_cell)-xi_forward_mat @ self.field_info['Phi_mat']),xi_E_inc)
            E_sca = self.field_info['Rec_mat'] @ J_trad
            ds_solution = {'J_trad': J_trad, 'E_sca': E_sca}
            np.save(ds_fwd_savedir, ds_solution)
            plot_J_figure(J_trad.squeeze(-1).cpu().numpy(), save_plot_path, args)

        # Create nerf model
        render_kwargs_train, render_kwargs_test, optimizer, start = self.create_isp_neuropretor(args)
        global_step = start
        if args.render_only:
            self.test(args, 'renderonly_{}_{:06d}.npy'.format('test' if args.render_test else 'path', start), render_kwargs_test, 'renderonly_{}_{:06d}.png'.format('test' if args.render_test else 'path', start), epsilon)
        if 'isp' in self.args.method:
            if self.args.epsilon_network == 'params':
                epsilon = render_kwargs_train['params_epsilon']
            elif self.args.epsilon_network == 'mlp':
                epsilon = render_kwargs_train['model_fwd_fn'](self.coords_eps, render_kwargs_train['model_epsilon']).squeeze(-1)
            elif self.args.epsilon_network == 'paramzation':
                epsilon = render_kwargs_train['model_epsilon'](render_kwargs_train['params_epsilon'])
        self.test(args, 'testset_000000.npy', render_kwargs_test, 'testset_000000.png', epsilon)

        N_iters = args.max_iter + 1
        loss_recorder = LossRecorder(args.exp_dir, loss_names=loss_list)
        print('Begin training...')
        start = start + 1
        last_regular_loss = 1
        for i in trange(start, N_iters):
            re = self.render(xi_E_inc, xi_forward_mat, self.field_info['Phi_mat'], self.field_info['Rec_mat'], epsilon, input_J=self.coords_inc, **render_kwargs_train)
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
                if 'fd' in args.method:
                    E_sca_evl_loss = (img2mse(re['Esca'].real, E_sca.real) + img2mse(re['Esca'].imag, E_sca.imag))/torch.mean(E_sca.real **2 + E_sca.imag **2)
                elif 'pd' in args.method:
                    E_sca_evl_loss = (img2mse(torch.abs(re['Esca']), torch.abs(E_sca)))/torch.mean(torch.abs(E_sca)**2)
                elif 'ad' in args.method:
                    E_sca_evl_loss = (img2mse(torch.angle(re['Esca']), torch.angle(E_sca)))/torch.mean(torch.angle(E_sca)**2)
                if 'param' in args.epsilon_network:
                    J_state_loss = (img2mse(re['J_state'].real, re['J'].real) + img2mse(re['J_state'].imag, re['J'].imag))/(re['norm_xi_E_inc'] + torch.finfo(torch.float32).eps)
                    loss = E_sca_evl_loss + J_state_loss + 0.01*regular_loss
                    # loss = (E_sca_evl_loss + J_state_loss)*((regular_loss+self.cell_area)/(last_regular_loss+self.cell_area))
                    # last_regular_loss = regular_loss.item()
                elif args.epsilon_network == 'mlp':
                    J_state_loss = (img2mse(re['J_state'].real, re['J'].real) + img2mse(re['J_state'].imag, re['J'].imag))/(re['norm_xi_E_inc'] + torch.finfo(torch.float64).eps)
                    if i<1000:
                        loss = E_sca_evl_loss + J_state_loss
                    else:
                        loss = E_sca_evl_loss + J_state_loss + 0.01*regular_loss
                current_losses = {'J_state_loss': J_state_loss.item(),'Esca_loss': E_sca_evl_loss.item(),
                                    'TV_loss': regular_loss.item(), 'BP_loss': loss.item()}
                loss_recorder.update(current_losses)

            loss.backward()
            optimizer['model'].step()
            if 'isp' in args.method and 'param' in args.epsilon_network:
                optimizer['params'].step()

            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            params_decay_steps = args.params_lrate_decay * 1000
            params_new_lrate = args.params_lrate * (decay_rate ** (global_step / params_decay_steps))
            new_lrate = {'model': new_lrate, 'params': params_new_lrate}
            for optimizer_name,optimizer_tmp in optimizer.items():
                if optimizer_tmp is not None:
                    for param_group in optimizer_tmp.param_groups:
                        param_group['lr'] = new_lrate[optimizer_name]

            if i % args.i_weights == 0:
                path = os.path.join(args.exp_dir, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'model_J_state_dict': render_kwargs_train['model_J'].state_dict(),
                    'optimizer_model_state_dict': optimizer['model'].state_dict(),
                    'model_epsilon_state_dict': render_kwargs_train['model_epsilon'].state_dict() if 'isp' in args.method and args.epsilon_network=='mlp' else None,
                    'optimizer_params_state_dict': optimizer['params'].state_dict() if 'isp' in args.method and 'param' in args.epsilon_network else None,
                }, path)
                print('Saved checkpoints at', path)

            if i % args.i_testset == 0 and i > 0:
                self.test(args, 'testset_{:06d}.npy'.format(i), render_kwargs_test, 'testset_{:06d}.png'.format(i), re['epsilon'])
                
            if i % args.i_print == 0:
                if args.method == 'fwd':
                    tqdm.write(f"[TRAIN] Iter: {i} J_state_loss: {J_state_loss.item()} E_sca_evl_loss: {E_sca_evl_loss.item()} J_evl_loss: {J_evl_loss.item()}")
                elif 'isp' in args.method:
                    tqdm.write(f"[TRAIN] Iter: {i} J_state_loss: {J_state_loss.item()} E_sca_evl_loss: {E_sca_evl_loss.item()} Regular_loss: {regular_loss.item()}")
            global_step += 1
        loss_recorder.plot_losses()
        loss_recorder.save_history()
        print('[TRAIN FINISH] model misfit:{}, data misfit:{}'.format(torch.sqrt(F.mse_loss(re['epsilon'].detach().cpu(), torch.from_numpy(np.load(args.params_path)).float()).item()/torch.mean(torch.from_numpy(np.load(args.params_path)).float()**2)),np.sqrt(E_sca_evl_loss.item())))

    def test(self, args, save_file, render_kwargs_test, save_plot_path=None, epsilon=None):
        testsavedir = os.path.join(args.exp_dir, save_file)
        with torch.no_grad():
            fn_test = render_kwargs_test['model_fwd_fn']
            output = fn_test(self.coords_inc, render_kwargs_test['model_J'])
            J_pred = output.squeeze(-1)
            Esca_pred = self.field_info['Rec_mat'] @ (J_pred[...,0]+1j*J_pred[...,1])
            ds_solution = {'J_pred': J_pred.cpu().numpy(), 'E_sca': Esca_pred.cpu().numpy()}
            np.save(testsavedir, ds_solution)
            if save_plot_path is not None:
                save_plot_path = os.path.join(args.exp_dir, save_plot_path)
                plot_J_figure(output.squeeze(-1).cpu().numpy(), save_plot_path, args)
            if 'isp' in args.method:
                save_plot_path_params = os.path.join(args.exp_dir, save_plot_path[:-4]+'_params.png')
                epsilon = epsilon.detach().cpu().numpy()
                plot_params_figure(epsilon, save_plot_path_params)
        print('Saved test set')
