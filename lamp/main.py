import os
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from . import trainer
from . import trainer_multifreqs

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./results/', help='where to store ckpts and logs')
    parser.add_argument("--params_path", type=str, default='./data/epsilon_gt.npy', help='params data path')
    parser.add_argument("--recdata_path", type=str, default='./data/E_sca.npy', help='receiver data path')
    parser.add_argument("--method", type=str, default='fwd', help='[fd-isp, fwd, pd-isp, ad-isp]')
    parser.add_argument("--J_network", type=str, default='single-mlp', help='[identify, single-mlp, multi-mlp]')
    parser.add_argument("--inc_wave", type=str, default='cir', help='[plane, cir]')
    parser.add_argument("--freq", type=str, default='0.4', help='frequency of the incident wave')
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--regularizer", type=str, default='tv_l1', help='choice from [tv_l1, tv_l2, mrtv, mdtv_l1, none]')
    parser.add_argument("--regularizer_decay", type=float, default=0.1, help='Decay of regularize weight')
    parser.add_argument("--regularizer_weight", type=float, default=0.1, help='Weight of regularize term')
    parser.add_argument("--epsilon_network", type=str, default='params', help='choice from [params, mlp, paramzation, paramzations, ft_paramz, proj_param]')
    parser.add_argument("--vae_model", type=str, default='vanilla_vae', help='choice from [vanilla_vae]')
    parser.add_argument("--vae_latent_dim", type=int, default=10, help='latent dimension for vae')
    parser.add_argument("--vae_ckpt_path", type=str, default=None, help='vae ckpt path')
    parser.add_argument("--vae_config_path", type=str, default=None, help='vae config path')
    parser.add_argument("--params_constraint", type=str, default=None, help='[none, positive, 12]')
    parser.add_argument("--max_params", type=float, default=2, help='Max permittivity in the DOI')
    parser.add_argument("--rm_same_points", action='store_true', help='remove same points of the tranceiver')
    parser.add_argument("--disable_projection", action='store_true', help='disable VAE projection when epsilon_network=proj_param')
    parser.add_argument("--projector_type", type=str, default='vae', choices=['vae', 'tv'], help='projection operator inside the alternating loop (R2-1a ablation): vae=LaMP latent-manifold projection, tv=TV-only proximal step on the pixel domain')
    parser.add_argument("--proj_tv_weight", type=float, default=0.05, help='gamma in Eq.(10); also the lambda of the TV-prox step when --projector_type=tv. Set to 0 for variant (i) "VAE projection without TV"')
    parser.add_argument("--proj_tv_weight_final", type=float, default=None, help='if set, linearly decay proj_tv_weight from --proj_tv_weight at first projection (iter=i_projection_start+i_projection) to this value at iter=max_iter. Mirrors DeepCSI regularizer_decay schedule for the projection inner-loop TV.')
    parser.add_argument("--proj_inner_steps", type=int, default=300, help='inner Adam steps for the TV-prox projector')
    parser.add_argument("--proj_inner_lr", type=float, default=1e-1, help='inner Adam learning rate for the TV-prox projector')
    parser.add_argument("--proj_inner_max", type=int, default=50, help='maximum inner Adam steps for each VAE latent-manifold projection')
    parser.add_argument("--proj_inner_min", type=int, default=0, help='minimum inner Adam steps before the VAE projection early-stop check is allowed')
    parser.add_argument("--proj_inner_tol", type=float, default=1e-2, help='relative rMSE tolerance for early stopping inside each VAE projection')
    parser.add_argument("--proj_z_init", type=str, default='ones',
                        choices=['encode', 'ones', 'zeros', 'zero', 'safe_encode'],
                        help='initial latent code for each LaMP projection event. encode=VAE-encode the intermediate epsilon; ones=latent of all-ones contrast; zeros/zero=literal zero latent; safe_encode=OOD-aware blend between encode(clipped epsilon) and encode(ones).')
    parser.add_argument("--proj_z_init_ood_tau", type=float, default=0.15,
                        help='OOD sensitivity for --proj_z_init safe_encode. Smaller values move the initialization toward encode(ones) more aggressively when epsilon leaves [1,max_params].')
    parser.add_argument("--proj_ood_metric", type=str, default='mean_sigma',
                        choices=['meanL1_both', 'rms_up', 'mean_sigma'],
                        help='OOD-score metric for safe_encode. meanL1_both=mean(|eps-clip|) over both sides (original; dilutes sparse-but-severe overflow). rms_up=sqrt(mean(relu(eps-max)^2)) upper-overflow L2. mean_sigma=mean(relu(eps-max))+k*std(relu(eps-max)) upper-overflow mean+sigma (std rescues sparse severe overflow without max single-pixel sensitivity).')
    parser.add_argument("--proj_ood_sigma_k", type=float, default=1.0,
                        help='sigma weight k for --proj_ood_metric mean_sigma.')
    parser.add_argument("--proj_z_init_alpha", type=float, default=None, help='if set, blend the GT-derived latent with the proj_z_init choice as alpha*z_GT + (1-alpha)*z_base. Default None uses the proj_z_init base only.')
    parser.add_argument("--i_projection_start", type=int, default=300, help='outer iteration after which projection is allowed; with i_projection=200 the default value 300 gives projections at {400,600,800,1000} (R2-Comment 3 ablation)')
    parser.add_argument("--i_projection_stop", type=int, default=1000000, help='outer iteration after which projection is disabled (default effectively infinity)')
    parser.add_argument("--enable_lazy_projection", action='store_true', help='With --epsilon_network=params, lazily inject VAE projection events at --i_projection cadence after --i_projection_start. The forward solve, optimizer, network and RNG state are identical to plain DeepCSI until the first projection trigger, at which point a VAEProjector is constructed on demand. Used to keep the DeepCSI/LaMP comparison in Fig. 3 strictly same-code-path (R2-C2).')
    parser.add_argument("--log_diagnostics", action='store_true', help='log per-iter chi_norm in addition to standard losses; combined with --save_metric this produces the convergence-diagnostic trace for R2-Comment 2')
    parser.add_argument("--mask_seed", type=int, default=None, help='deterministic seed for sparse Tx/Rx subsampling pattern (R2-Comment 2 multi-mask statistics); None = use the regular --seed for both noise and mask')

    parser.add_argument("--L_doi", type=float, default=2, help='DOI length')
    parser.add_argument("--R_t", type=float, default=3, help='Transmitter radius')
    parser.add_argument("--R_r", type=float, default=3.5, help='Receiver radius')
    parser.add_argument("--tx_angle_range", type=float, nargs=2, default=[0.0, 360.0],
                        help='[start_deg, end_deg) angular span for transmitters')
    parser.add_argument("--rx_angle_range", type=float, nargs=2, default=[0.0, 360.0],
                        help='[start_deg, end_deg) angular span for receivers')
    parser.add_argument("--N_rec", type=int, default=32, help='Nb. of Receiver')
    parser.add_argument("--N_rec_data", type=int, default=241, help='Nb. of Receiver in raw Fresnel data (before subsampling)')
    parser.add_argument("--N_rec_use", type=int, default=None, help='Nb. of Receiver to use after subsampling (<= N_rec_data)')
    parser.add_argument("--double_rec", action='store_true', help='Use double receiver at different circles')
    parser.add_argument("--N_inc", type=int, default=16, help='Nb. of Incidence')
    parser.add_argument("--N_inc_data", type=int, default=None, help='Nb. of Incidence in raw Fresnel data (before subsampling); default follows N_inc')
    parser.add_argument("--N_inc_use", type=int, default=None, help='Nb. of Incidence to use after subsampling (<= N_inc_data)')
    parser.add_argument("--grid_num", type=int, default=64, help='Nb. of grid points')

    # training options
    parser.add_argument("--max_iter", type=int, default=2000, help='max iterations to train in pwd and isp')
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
    parser.add_argument("--lrate", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--lrate_decay", type=float, default=1, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--params_lrate", type=float, default=1e-1, help='learning rate for optimized params')
    parser.add_argument("--params_lrate_decay", type=float, default=1, help='exponential learning rate decay (in 1000 steps) for optimized params')
    parser.add_argument("--vae_lrate", type=float, default=1e-4, help='learning rate for optimized params')
    parser.add_argument("--vae_lrate_decay", type=float, default=1, help='exponential learning rate decay (in 1000 steps) for optimized params')
    parser.add_argument("--ft_path", type=str, default=None, help='specific weights npy file to reload for coarse network')
    parser.add_argument("--i_projection", type=int, default=300, help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--i_embed", type=int, default=0, help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding')
    parser.add_argument("--render_only", action='store_true', help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--reload_weight", action='store_true', help='reload weights from previous ckpt')
    
    # logging/saving options
    parser.add_argument("--i_print", type=int, default=1, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=5000, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=100, help='frequency of testset saving')
    parser.add_argument("--i_regularizer", type=int, default=1, help='frequency of regularizer update')
    parser.add_argument("--noise_type", type=str, default='gaussion', help='noise type added in the measurement data')
    parser.add_argument("--noise_ratio", type=float, default=0.0, help='noise_ratio')
    parser.add_argument("--save_noisy_data", type=str, default=None, help='if set, save the noisy E_sca tensor to this .npy path so other methods can reuse the identical noise realization (RNG alignment across methods)')
    parser.add_argument("--load_noisy_data", type=str, default=None, help='if set, skip internal noise generation and load the noisy E_sca tensor from this .npy path (RNG alignment across methods)')
    parser.add_argument("--result_file", type=str, default=None, help='record the result of inversion')
    parser.add_argument("--save_metric", action='store_true', help='Save the metric of inversion with loss')

    return parser


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    N_inc = args.N_inc_use if args.N_inc_use is not None else args.N_inc
    N_rec = args.N_rec_use if args.N_rec_use is not None else args.N_rec
    args.expname = (args.expname + '_{}'.format(args.params_path.split('/')[-1].split('_')[-1].split('.')[0]) + '_{}'.format(args.method) 
                    + '_{}'.format(args.inc_wave) +'_{}*{}'.format(N_inc,N_rec) +'_{}'.format('double_rec' if args.double_rec else 'single_rec') + '_{}GHz'.format(args.freq)) + '_{}'.format(args.J_network) + '_{}*{}'.format(args.netdepth, args.netwidth)
    if 'isp' in args.method:
        args.expname += '_{}'.format(args.epsilon_network)
        if args.epsilon_network == 'proj_param':
            args.expname += '{}'.format(args.i_projection)
        if args.epsilon_network in ['paramzation', 'paramzations', 'ft_paramz', 'proj_param']:
            args.expname += '_{}'.format(args.vae_model.replace("_",""))+ '_latent{}'.format(args.vae_latent_dim)
        args.expname += '_{}'.format(args.regularizer)+'{}'.format(args.regularizer_weight)+ '_nsr{}'.format(args.noise_ratio)
    args.exp_dir = os.path.join(args.basedir, args.expname)
    os.makedirs(args.exp_dir, exist_ok=True)
    
    for arg in sorted(vars(args)):
        attr = getattr(args, arg)
        print('{} = {}'.format(arg, attr))
    
    if 'TwinDiel' in args.recdata_path:
        args.N_inc = 18
    
    if ',' not in args.freq:
        pinns_isp = trainer.PINNsFwdIsp2D(args)
        pinns_isp.train(args)
    else:
        pinns_isp = trainer_multifreqs.PINNsFwdIsp2D(args)
        pinns_isp.train(args)
