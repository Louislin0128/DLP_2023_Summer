#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from dataloader import iclevrLoader
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from evaluator import evaluation_model
from diffusers import DDPMScheduler, UNet2DModel

class ClassConditionedUnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.batch_size      = args.batch_size
        self.mse_criterion   = nn.MSELoss()
        self.optim           = optim.AdamW(self.parameters(), lr=self.args.lr)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
        
        self.denormalize = transforms.Compose([
            transforms.Normalize((0, 0, 0), (2, 2, 2)),
            transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1))
        ])
    
        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=128,          # the target image resolution
            in_channels=27,           # Additional input channels for class cond.
            out_channels=3,           # the number of output channels
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64), 
            down_block_types=( 
                "DownBlock2D",        # a regular ResNet downsampling block
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ), 
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",          # a regular ResNet upsampling block
              ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_labels):
        # Shape of x:
        bs, ch, w, h = x.shape

        class_labels = class_labels.view(bs, class_labels.shape[1], 1, 1).expand(bs, class_labels.shape[1], w, h)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_labels), 1) 

        # Feed this to the unet alongside the timestep and return the prediction
        return self.model(net_input, t).sample 


    def train(self):
        # Keeping a record of the losses for later viewing
        losses = []

        # The training loop
        for epoch in range(self.args.num_epoch):
            train_dataloader = DataLoader(iclevrLoader('iclevr\\', 'train'), batch_size = self.batch_size, shuffle = True)
            
            for data, label in tqdm(train_dataloader):
                data = data.to(self.args.device)
                label = label.to(self.args.device)
                noise = torch.randn_like(data)
                timesteps = torch.randint(0, 999, (data.shape[0],)).long().to(device)
                noisy_x = self.noise_scheduler.add_noise(data, noise, timesteps)

                # Get the model prediction
                pred = self.model(noisy_x, timesteps, label) # Note that we pass in the labels y

                # Calculate the loss
                loss = self.mse_criterion(pred, noise) # How close is the output to the noise

                # Backprop and update the params:
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.)
                self.optim.step()

                # Store the loss for later
                losses.append(loss.item())

            # Print our the average of the last 100 loss values to get an idea of progress:
            avg_loss = sum(losses[-100:])/100
            print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')

        # View the loss curve
        plt.plot(losses)
        self.save(os.path.join(self.args.save_root, f"epoch={self.args.num_epoch}.ckpt"))
    
    def test(self):
#         valid_dataloader = DataLoader(iclevrLoader('iclevr\\', 'valid'), batch_size = self.batch_size, shuffle = False)
        test_dataloader = DataLoader(iclevrLoader('iclevr\\', 'test'), batch_size = self.batch_size, shuffle = False)
        
        self.model.eval()
        for label in test_dataloader:
            x = torch.randn(len(label), 3, 64, 64).to(self.args.device)
            label = torch.from_numpy(np.array(label)).to(self.args.device) 
            num = 0
            # Sampling loop
            for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                
                # Get model pred
                with torch.no_grad():
                    residual = self.model(x, t, label)  # Again, note that we pass in our labels y

                # Update sample with step
                x = self.noise_scheduler.step(residual, t, x).prev_sample
                
            evaluator = evaluation_model()
            acc = evaluator.eval(x, label)
            print("Testing accuracy: ", acc)
            
            # Show the results
            fig = plt.subplots(1, 1, figsize=(12, 12))
            img = make_grid(self.denormalize(x), nrow = 8)
            im = torchvision.transforms.ToPILImage()(img)
            save_image(im, "output.png")
        
        
    """模型儲存"""
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer" : self.state_dict(),  
            "lr"        : self.state_dict(),
        }, path)
        print(f"save ckpt to {path}")
        
    """模型載入"""
    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            
            self.optim      = optim.AdamW(self.parameters(), lr=self.args.lr)
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    
def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = ClassConditionedUnet(args).to(device)
    
    if args.test:
        model.eval()
    else:
        model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=128)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="AdamW")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_epoch',     type=int, default=1,     help="number of total epoch")
    
    args = parser.parse_args()
    
    main(args)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




