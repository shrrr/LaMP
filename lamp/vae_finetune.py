# https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/README.md
from math import e
import os
from omegaconf import OmegaConf
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
from PIL import Image
from argparse import ArgumentParser
import lpips
from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma
from contextlib import contextmanager
from torchmetrics.image import StructuralSimilarityIndexMeasure
from matplotlib import cm
from pytorch_lightning.loggers import CSVLogger

torch.cuda.empty_cache()

class FinetuneFaceData(Dataset):
    def __init__(self, data_dir:str, 
                 img_list: list,
                 size:int=384, 
                 channels:int=1,
                 max_params:int=2):
        self.data_dir = data_dir
        self.img_list = img_list
        self.size = size
        self.channels = channels
        self.max_params = max_params
        mean = [0.5] * channels
        std = [0.5] * channels
        self.transform = transforms.Compose([
            self.matrix_to_tensor,
            transforms.Normalize(mean=mean, std=std),
        ])
    def matrix_to_tensor(self, matrix):
        if matrix.shape[0] != self.size or matrix.shape[1] != self.size:
            matrix = np.resize(matrix, (self.size, self.size))
        if matrix.ndim == 2:
            matrix = matrix[np.newaxis, :, :]
        elif matrix.ndim == 3 and matrix.shape[0] != self.channels:
            matrix = matrix.transpose(2, 0, 1)
        matrix = (matrix-1)/(self.max_params-1)
        return torch.FloatTensor(matrix)
    
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.img_list[idx])
        image = np.load(img_name)
        return self.transform(image), self.img_list[idx]
    
class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, 
                 batch_size=64, 
                 val_size=0.1,
                 size=384,
                 channels=1,
                 max_params=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.size = size
        self.channels = channels
        self.max_params = max_params
        self.setup('fit')
    def setup(self, stage):
        all_images = sorted([u for u in os.listdir(self.data_dir) if u.endswith(".npy")])
        random.shuffle(all_images)
        train_size = int((1-self.val_size)*len(all_images))
        train_images = all_images[:train_size]
        val_images = all_images[train_size:] 
        self.train_ds = FinetuneFaceData(self.data_dir,  train_images, self.size, self.channels, self.max_params)
        self.val_ds = FinetuneFaceData(self.data_dir,  val_images, self.size, self.channels, self.max_params)
        print(f"Train size: {len(self.train_ds)}, Val size: {len(self.val_ds)}")
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

class FinetuneVAE(pl.LightningModule):
    def __init__(self, 
                 kl_weight=0.1, 
                 lpips_loss_weight=0.1,
                 lr=1e-4, 
                 momentum=0.9, 
                 weight_decay=5e-4,
                 optim='sgd',
                 vae_finetune=True,
                 vae_config=None,
                 vae_weights=None,
                 device=torch.device('cuda'),
                 ema_decay=0.999,
                 precision=32,
                 log_dir=None,
                 channels=1,
                 max_params=2,
                 **unused_kwargs):
        super().__init__()
        self.kl_weight = kl_weight
        self.lpips_loss_weight = lpips_loss_weight
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_loss_fn.eval()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optim = optim
        self.channels = channels
        self.max_params = max_params
        self.model =  instantiate_from_config(vae_config)
        if channels == 1 and vae_weights is not None:
            # 这里需要处理权重适配问题，具体实现取决于模型结构
            self._adapt_weights_for_grayscale(vae_weights)
        if vae_finetune:
            self.model.load_state_dict(vae_weights, strict=False)
        self.model.train()
        self.precision = precision
        self.log_dir = log_dir
        self.log_one_batch = False
        self.use_ema = ema_decay > 0
        if self.use_ema :    
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self.model, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.validation_step_outputs = []
        
    def _adapt_weights_for_grayscale(self, weights):
        if 'encoder.conv_in.weight' in weights:
            conv_weight = weights['encoder.conv_in.weight']
            if conv_weight.size(1) == 3 and self.channels == 1:
                weights['encoder.conv_in.weight'] = conv_weight.mean(dim=1, keepdim=True)
        if 'decoder.conv_out.weight' in weights:
            conv_weight = weights['decoder.conv_out.weight']
            conv_bias = weights['decoder.conv_out.bias']
            if conv_weight.size(0) == 3 and self.channels == 1:
                weights['decoder.conv_out.weight'] = conv_weight.mean(dim=0, keepdim=True)
                weights['decoder.conv_out.bias'] = conv_bias.mean(dim=0, keepdim=True)
                
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Assuming the DataModule is attached to the Trainer and accessible
            self.train_ds = self.trainer.datamodule.train_ds
            self.val_ds = self.trainer.datamodule.val_ds
            print("Warning: The setup method is called")
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        target, _ = batch
        if self.precision == 16:
            target = target.half()
        posterior = self.model.encode(target)
        z = posterior.sample()
        pred = self.model.decode(z)
        kl_loss = posterior.kl()
        kl_loss = kl_loss.mean() 
        rec_loss = torch.abs(target.contiguous() - pred.contiguous())
        # if self.current_epoch < self.trainer.max_epochs // 3 * 2:
        #     rec_loss = rec_loss.mean() * rec_loss.size(1)
        # else:
        #     rec_loss = rec_loss.pow(2).mean() * rec_loss.size(1)
        rec_loss = rec_loss.mean() * rec_loss.size(1)
        if self.channels == 1:
            target_3ch = target.repeat(1, 3, 1, 1)
            pred_3ch = pred.repeat(1, 3, 1, 1)
            lpips_loss = self.lpips_loss_fn(pred_3ch, target_3ch).mean()
        else:
            lpips_loss = self.lpips_loss_fn(pred, target).mean()
        loss = rec_loss + self.lpips_loss_weight * lpips_loss + self.kl_weight * kl_loss
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('rec_loss', rec_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('lpips_loss', lpips_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('kl_loss', kl_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss
    def configure_optimizers(self):
        if self.optim == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        return optimizer
    def validation_step(self, batch, batch_idx):  
        target, name = batch
        if self.precision == 16:
            target = target.half()
        posterior = self.model.encode(target)
        z = posterior.mode()
        pred = self.model.decode(z)
        kl_loss = posterior.kl()
        kl_loss = kl_loss.mean() # torch.sum(kl_loss) / kl_loss.shape[0]
        rec_loss = torch.abs(target.contiguous() - pred.contiguous())
        rec_loss = rec_loss.mean() # torch.sum(rec_loss) / (rec_loss.shape[0] *  rec_loss.shape[2] * rec_loss.shape[3])
        mse = F.mse_loss(pred, target)
        ssim = self.ssim(pred, target)
        if self.channels == 1:
            target_3ch = target.repeat(1, 3, 1, 1)
            pred_3ch = pred.repeat(1, 3, 1, 1)
            lpips_loss = self.lpips_loss_fn(pred_3ch, target_3ch).mean()
        else:
            lpips_loss = self.lpips_loss_fn(pred, target).mean()
        loss = rec_loss + self.lpips_loss_weight * lpips_loss + self.kl_weight * kl_loss
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.append({
            'val_loss': loss, 
            "rec_loss": rec_loss, 
            "lpips_loss": lpips_loss, 
            "kl_loss": kl_loss,
            "ssim": ssim, 
            "mse": mse
        })
        self.log_images(target, pred, name)
        return {'val_loss': loss, "rec_loss": rec_loss, "lpips_loss": lpips_loss, "kl_loss": kl_loss, "ssim": ssim, "mse": mse}
    def log_images(self, input, output, names):
        if self.log_one_batch: 
            return 
        for img1, img2, name in  zip(input, output, names):
            os.makedirs(self.log_dir + "/" + str(self.current_epoch), exist_ok=True)
            if self.channels == 1:
                img1 = img1.cpu().detach().numpy().squeeze(0)
                img2 = img2.cpu().detach().numpy().squeeze(0)
                img1_norm = ((img1*0.5)+0.5)
                img2_norm = ((img2*0.5)+0.5)
                img1 = (img1_norm * (self.max_params-1)) + 1
                img2 = (img2_norm * (self.max_params-1)) + 1
                diff = abs(img1 - img2)
                img = np.concatenate([img1, img2, diff], axis=1)
                img_norm = np.concatenate([img1_norm, img2_norm, abs(img1_norm - img2_norm)], axis=1)
                np.save(os.path.join(self.log_dir, str(self.current_epoch), name), img)
                colored_img = cm.viridis(img_norm)  # 应用colormap
                colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)  # 保留RGB通道，转换为0-255
                pil_img = Image.fromarray(colored_img)
                pil_img.save(os.path.join(self.log_dir, str(self.current_epoch), name.replace('.npy', '.png')))
            else:
                img1 = img1.cpu().detach().numpy().transpose(1, 2, 0)
                img2 = img2.cpu().detach().numpy().transpose(1, 2, 0)
                img1 = (img1 + 1) / 2
                img2 = (img2 + 1) / 2
                diff = abs(img1 - img2)
                img = np.concatenate([img1, img2, diff], axis=1)
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(self.log_dir, str(self.current_epoch), name))
        self.log_one_batch = True
    def on_train_epoch_end(self):
        if self.use_ema:
            self.model_ema(self.model)
            self.model_ema.copy_to(self.model)
        if self.current_epoch == self.trainer.max_epochs // 3 * 2:
            self.lpips_loss_weight = self.lpips_loss_weight * 0.1
    def on_validation_epoch_end(self):
        self.log_one_batch = False

        outputs = self.validation_step_outputs
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        rec_loss = torch.stack([x['rec_loss'] for x in outputs]).mean()
        lpips_loss = torch.stack([x['lpips_loss'] for x in outputs]).mean()
        kl_loss = torch.stack([x['kl_loss'] for x in outputs]).mean()
        ssim = torch.stack([x['ssim'] for x in outputs]).mean()
        mse = torch.stack([x['mse'] for x in outputs]).mean()
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_rec_loss', rec_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_lpips_loss', lpips_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_ssim', ssim, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_mse', mse, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.validation_step_outputs = []

class SDVAEFT(nn.Module):
    def __init__(self, vae_config_path='/root/shared-nvme/PINNs-IE/PINNs-ISP-IncompleteData/code/vae_config.yaml', 
                 precision=32, max_params=2, mean=0.5, std=0.5,
                 **unused_kwargs):
        super().__init__()
        self.precision = precision
        self.max_params = max_params
        self.mean = mean
        self.std = std
        config = OmegaConf.load(vae_config_path)
        self.model = instantiate_from_config(config.model)
    
    def load_state_dict(self, state_dict, strict=False):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        vae_weight = dict(state_dict)
        for k in state_dict.keys():
            if "model" in k:
                vae_weight[k.replace("model.", "")] = state_dict[k]
        # model_keys = list(self.model.state_dict().keys())
        # matched_weights = {}
        # missing_keys = []
        # for k in model_keys:
        #     if k in vae_weight:
        #         if vae_weight[k].shape == self.model.state_dict()[k].shape:
        #             matched_weights[k] = vae_weight[k]
        #         else:
        #             print(f"形状不匹配: {k}, 预训练: {vae_weight[k].shape}, 模型: {self.model.state_dict()[k].shape}")
        #     else:
        #         missing_keys.append(k)
        self.model.load_state_dict(vae_weight, strict)
    
    def normalize(self, x):
        x = (x-1)/(self.max_params-1)
        x = (x-self.mean)/self.std
        return x
    
    def denormalize(self, x):
        x = x*self.std + self.mean
        x = x*(self.max_params-1) + 1
        return x
    
    def forward(self, x):
        mean, std = self.encode(x)
        rec = self.decode(mean)
        return rec, mean, std
    
    def encode(self, x):
        x = self.normalize(x)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if self.precision == 16:
            x = x.half()
        posterior = self.model.encode(x)
        if hasattr(posterior, 'mean') and hasattr(posterior, 'logvar'):
            return posterior.mean, posterior.logvar
        elif hasattr(posterior, 'mode'):
            return posterior.mode(), None
        else:
            return posterior.sample(), None
        
    def decode(self, z):
        if z.dim() == 2:
            h = self.trainer.datamodule.size // 8
            w = self.trainer.datamodule.size // 8
            c = 4
            z = z.view(-1, c, h, w)
        decoded = self.model.decode(z)
        decoded = self.denormalize(decoded)
        return decoded

def get_vae_weights(input_path):
    pretrained_weights = torch.load(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']
    vae_weight = dict(pretrained_weights)
    for k in pretrained_weights.keys():
        if "first_stage_model" in k:
            vae_weight[k.replace("first_stage_model.", "")] = pretrained_weights[k]
    return vae_weight

def argument_inputs():
    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./',
                        help='The directory that contains the images, including original folder and the emotion folders.')
    parser.add_argument('--data_dir', type=str, default='./dataset/',
                        help='The directory that contains the images, including original folder and the emotion folders.')
    parser.add_argument('--ema_decay',  type=float, default=0.99 ,help="Use use_ema") 

    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/',
                        help='The directory that contains the pretrained VAE checkpoint folders.')
    parser.add_argument('--ckpt_name', type=str, default='kl_f8',
                        help='The ckpt name that need to be finetuned.')
     
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32])
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--channels', type=int, default=1, 
                        help='图像通道数(1表示灰度图,3表示彩色图)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_params', type=float, default=2.0)

    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--kl_weight', type=float, default=1.)
    parser.add_argument('--lpips_loss_weight', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--output_dir', type=str, 
                        default='./vae_finetune',)
    parser.add_argument('--note', type=str, 
                        default='',)
    args =  parser.parse_args()
    args.devices = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    args.strategy = "ddp" #"ddp"
    return args


if __name__ == '__main__':
    args = argument_inputs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_names = f"size_({args.image_size})_val({args.val_size})_ema({args.ema_decay})_bs({args.batch_size})_lr({args.lr})_epochs({args.num_epochs})_kl({args.kl_weight})_lpips({args.lpips_loss_weight})_{args.note}"
    log_dir = f"{args.output_dir}/{args.ckpt_name}/{file_names}"
    os.makedirs(log_dir, exist_ok=True)
    logger = CSVLogger(save_dir=log_dir, name="my_logs")
    config = OmegaConf.load(os.path.join(args.ckpt_dir, args.ckpt_name + ".yaml"))
    if args.channels == 1:
        if 'params' in config.model and 'in_channels' in config.model.params.ddconfig:
            config.model.params.ddconfig.in_channels = 1
        if 'params' in config.model and 'out_ch' in config.model.params.ddconfig:
            config.model.params.ddconfig.out_ch = 1
    vae_config = config.model
    input_path = os.path.join(args.ckpt_dir, args.ckpt_name + ".ckpt")
    vae_weight = get_vae_weights(input_path)
    data_module = DataModule(args.data_dir, 
                             batch_size=args.batch_size, 
                             val_size=0.1,
                             size=args.image_size,
                             channels=args.channels,
                             max_params=args.max_params)
    
    vae_finetune = False
    model = FinetuneVAE(vae_finetune=vae_finetune,
                        vae_config=vae_config, 
                        vae_weights=vae_weight, 
                        kl_weight=args.kl_weight, 
                        lpips_loss_weight=args.kl_weight,
                        lr=args.lr, 
                        device=device,
                        log_dir=log_dir,
                        ema_decay=args.ema_decay,
                        channels=args.channels)

    trainer = Trainer(min_epochs=1, 
                      max_epochs=args.num_epochs, 
                      precision=args.precision,
                      strategy=args.strategy, 
                      devices=args.devices, 
                      num_sanity_val_steps=1 if args.val_size > 0 else 0,
                      default_root_dir=log_dir,
                      logger=logger,)

    trainer.fit(model, datamodule=data_module)
    torch.save(model.model.state_dict(), f"{log_dir}/last_model.pth")
