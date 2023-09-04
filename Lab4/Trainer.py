import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

"""PSNR for torch tensor"""
def Generate_PSNR(imgs1, imgs2, data_range=1.):
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

""" KLD Loss func """
def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size  
    return KLD

""" 要實作的地方 """
class kl_annealing():
    def __init__(self, args, current_epoch = 0):
    
        self.current_epoch = current_epoch
        self.kl_anneal_type = args.kl_anneal_type
        self.beta = 0.0
        
    def update(self):
        self.current_epoch += 1
        
        if self.kl_anneal_type == "Monotonic":
            self.beta = min(1.0, self.beta + 1.0 / args.kl_anneal_cycle)
            
        elif self.kl_anneal_type == 'Cyclical':
            self.beta = self.frange_cycle_linear(args.num_epoch, start = 0.0, stop = 1.0, 
                                                 n_cycle = 4.0, ratio = args.kl_anneal_ratio)
        else:
            self.beta = 1.0    # Default, Without KL annealing

    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start = 0.0, stop = 1.0,  n_cycle = 10, ratio = 1.0):
        period = n_iter / n_cycle
        step = self.current_epoch % period
        
        return start + step * (stop - start) / (period * ratio) if step < (period * ratio) else stop

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator  = Generator(input_nc=args.D_out_dim, output_nc=3)
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch = 0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size   = args.batch_size
    
    """ 需要實作的地方 """
    def forward(self, img, label, adapt_TeacherForcing):
        # Initialize frame
        reconstructed_img = [img[:,0]]
        
        for i in range(1, img.size(1)):
            if adapt_TeacherForcing:
                previous_frames = img[:, i-1]  # Use ground truth image for teacher forcing
            else:
                # Generate image using the model's own previous reconstruction
                previous_frames = reconstructed_img[-1]
            
            # Image transformation
            previous_features = self.frame_transformation(previous_frames)
            ground_truth_features = self.frame_transformation(img[:, i])
        
            # Label transformation
            label_features = self.label_transformation(label[:, i])
        
            # Predict latent parameters (sample latent variable)
            z, mu, logvar = self.Gaussian_Predictor(ground_truth_features, label_features)

            # Fusion for decoder input
            decoder_input = self.Decoder_Fusion(previous_features, label_features, z)

            # Generate output
            generated_output = self.Generator(decoder_input)
            reconstructed_img.append(generated_output)
        
        reconstructed_img = stack(reconstructed_img, dim = 1)
        
        return reconstructed_img, mu, logvar       
    
    def training_stage(self):
        train_Loss, tf_adapt, tfr_arr, beta_record, val_Loss, valid_psnr = [],[],[],[],[],[]
        num = []
        
        for i in range(self.args.num_epoch):
            tot_loss = 0.0
            num.append(i+1)
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            for (img, label) in (pbar := tqdm(train_loader, ncols = 120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                tot_loss += float(loss.detach().cpu())
                beta = self.kl_annealing.get_beta()
                
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            tot_loss /= len(train_loader.dataset)
            print(f"training loss: ", tot_loss)
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            
            train_Loss.append(round(tot_loss, 5))
            tf_adapt.append(adapt_TeacherForcing)
            tfr_arr.append(round(self.tfr, 10))
            beta_record.append(beta)      
            
            """ eval stage """
            epoch_psnr, vloss = self.eval() 
            val_Loss.append(round(vloss, 5))
            valid_psnr.append(round(epoch_psnr, 4))
                                  
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
        
        df = {'Epoch': num, 'train_loss': train_Loss, 'teacherForce(T/F)': tf_adapt, 'teacher_forcing_ratio': tfr_arr, 
              'beta': beta_record, 'valid_loss': val_Loss, 'valid_PSNR': valid_psnr}
        
        new_df = pd.DataFrame(df)
        new_df.to_csv(self.args.save_root + '\\' + self.args.kl_anneal_type + ".csv", index = False)        
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        psnr_list = []
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, generated_frame = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            for i in range(1, len(generated_frame)):
                PSNR = Generate_PSNR(img[0][i], generated_frame[i])
                psnr_list.append(PSNR.item())
            
        return sum(psnr_list)/(len(psnr_list)-1), float(loss.detach().cpu()) 
    
    """ training過程 """
    def training_one_step(self, img, label, adapt_TeacherForcing):
        self.optim.zero_grad()
       
        # Forward pass
        reconstructed_img, mu, logvar = self.forward(img, label, adapt_TeacherForcing)
        
        # Calculate reconstruction loss (MSE)
        mse_loss = self.mse_criterion(reconstructed_img, img)

        # Calculate KL divergence
        kl_loss = kl_criterion(mu, logvar, img.size(0))
        
        # Apply annealed KL weight
        total_loss = mse_loss + self.kl_annealing.get_beta() * kl_loss

        # Backpropagation
        total_loss.backward()
        self.optimizer_step()

        return total_loss
    
    """ validating過程 """
    def val_one_step(self, img, label):
        # Forward pass
        reconstructed_img, mu, logvar = self.forward(img, label, False)

        # Calculate reconstruction loss (MSE)
        mse_loss = self.mse_criterion(reconstructed_img, img)

        # Calculate KL divergence
        kl_loss = kl_criterion(mu, logvar, img.size(0))

        total_loss = mse_loss + self.kl_annealing.get_beta() * kl_loss

        return total_loss, reconstructed_img.squeeze()
    
    
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
        
    """載入training dataset"""
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)

        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset, batch_size = self.batch_size, num_workers = self.args.num_workers,
                                  drop_last = True, shuffle = False)  
        return train_loader
    
    """載入validating dataset"""
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        
        dataset = Dataset_Dance(root = self.args.DR, transform = transform, mode = 'val', video_len = self.val_vi_len, partial = 1.0)  
        val_loader = DataLoader(dataset, batch_size = 1, num_workers = self.args.num_workers, drop_last = True, shuffle = False)  
        
        return val_loader
    
    """ 要實作的地方 """
    def teacher_forcing_ratio_update(self):
        if self.current_epoch % self.tfr_sde == 0:
            self.tfr = max(0.1, (self.tfr - self.tfr_d_step * 2))
            
            # modify tfr when every decay ==> tfr = tfr - 0.1
    
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss = float(loss), refresh = False)
        pbar.refresh()
    
    
    """模型儲存"""
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer" : self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       : self.tfr,
            "last_epoch": self.current_epoch,
            "beta"      : self.kl_annealing.get_beta()
        }, path)
        print(f"save ckpt to {path}")
    
    
    """模型載入"""
    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']
    
    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    
    if args.test:
        model.eval()
    else:
        model.training_stage()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=1,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,   default=None, help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Monotonic
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,              help="")
    
    args = parser.parse_args()
    
    main(args)
