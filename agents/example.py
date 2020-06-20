import os
import os.path as osp
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from datasets.berkeley import BerkeleyDataset
from models.cyclegan import CycleGAN

__all__ = [ "BerkeleyAgent" ]

class BerkeleyAgent:

    def __init__(self, config):
        ### Configuration option
        self.config = config

        ### Training dataset
        transform = transforms.Compose([
            transforms.Resize(config['dataset']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset = BerkeleyDataset(root="download",
                                name=config['dataset']['name'],
                                transform=transform,
                                train=True)

        self.dataloader = DataLoader(dataset,
                                    batch_size=config['dataset']['batch_size'],
                                    shuffle=True)

        ### Training environment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### Models to train
        self.cyclegan = CycleGAN(channels_img=config['cyclegan']['channels_img'],
                                features_g=config['cyclegan']['features_g'],
                                blocks_g=config['cyclegan']['blocks_g'],
                                features_d=config['cyclegan']['features_d'],
                                pool_size=config['cyclegan']['pool_size'])
        self.cyclegan.to(self.device)
        self.cyclegan.train()

        ### Optimizer & Loss function
        self.optimizerG = optim.Adam(itertools.chain(self.cyclegan.netG_AB.parameters(),
                                                    self.cyclegan.netG_BA.parameters()),
                                    lr=config['train']['lr'],
                                    betas=(0.5, 0.999))
        self.optimizerD = optim.Adam(itertools.chain(self.cyclegan.netD_A.parameters(),
                                                    self.cyclegan.netD_B.parameters()),
                                    lr=config['train']['lr'],
                                    betas=(0.5, 0.999))
        self.criterionG = nn.MSELoss().to(self.device)
        self.criterionD = nn.MSELoss().to(self.device)
        self.criterionCycle = nn.L1Loss().to(self.device)

        ### LR Schedular
        optimizers = [ self.optimizerD, self.optimizerG ]
        def lambda_rule(epoch):
            lr_l = 1.0 - (max(0, epoch+1) / float(self.config['train']['n_epochs'] + 1))
            return lr_l
        self.schedulars = [ LambdaLR(optimizer, lr_lambda=lambda_rule)
                            for optimizer in optimizers ]

        ### Validation & Tensorboard
        self.fixed_real_A = torch.stack([dataset[i][0] for i in range(5)]).to(self.device)
        self.fixed_real_B = torch.stack([dataset[i][1] for i in range(5)]).to(self.device)
        self.writer_ABA = SummaryWriter(osp.join(config['tensorboard']['dir'], 'domainABA'))
        self.writer_BAB = SummaryWriter(osp.join(config['tensorboard']['dir'], 'domainBAB'))

    def train(self):
        for epoch in range(self.config['train']['n_epochs']):
            print("Current LR:", self.optimizerG.param_groups[0]['lr'])
            self.train_one_epoch(epoch)
            for schedular in self.schedulars:
                schedular.step()
            print("="*20)

    def train_one_epoch(self, epoch):
        for batch_idx, (real_A, real_B) in enumerate(self.dataloader):
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            real_label = torch.ones(len(real_A)).to(self.device)
            fake_label = torch.zeros(len(real_A)).to(self.device)

            ### Forward & Cycle pass
            self.cyclegan.train()
            fake_B = self.cyclegan.netG_AB(real_A) # A -> B
            fake_A = self.cyclegan.netG_BA(real_B) # B -> A
            rec_B = self.cyclegan.netG_AB(fake_A)  # A -> B
            rec_A = self.cyclegan.netG_BA(fake_B)  # B -> A

            ### Train Discriminator (netD_A, netD_B)
            self.optimizerD.zero_grad()
            for net in [self.cyclegan.netD_A, self.cyclegan.netD_B]:
                for param in net.parameters():
                    param.requires_grad = True

            # Discriminator loss (same distribution): (D(a)-1)^2
            real_A_label = self.cyclegan.netD_A(real_A).reshape(-1)
            real_B_label = self.cyclegan.netD_B(real_B).reshape(-1)
            real_A_lossD = self.criterionD(real_A_label, real_label)
            real_B_lossD = self.criterionD(real_B_label, real_label)

            # Discriminator loss (different distribution): D(a)^2
            fake_A = self.cyclegan.image_pool_A.query(fake_A)
            fake_B = self.cyclegan.image_pool_B.query(fake_B)
            fake_A_label = self.cyclegan.netD_A(fake_A.detach()).reshape(-1)
            fake_B_label = self.cyclegan.netD_B(fake_B.detach()).reshape(-1)
            fake_A_lossD = self.criterionD(fake_A_label, fake_label)
            fake_B_lossD = self.criterionD(fake_B_label, fake_label)

            # Final loss
            A_lossD = (real_A_lossD + fake_A_lossD)*0.5
            B_lossD = (real_B_lossD + fake_B_lossD)*0.5

            # Update discriminator
            A_lossD.backward()
            B_lossD.backward()
            self.optimizerD.step()

            ### Train Generator
            self.optimizerG.zero_grad()
            for net in [self.cyclegan.netD_A, self.cyclegan.netD_B]:
                for param in net.parameters():
                    param.requires_grad = False

            # Generator loss: (D(G(a))-1)^2
            fake_A_label = self.cyclegan.netD_A(fake_A).reshape(-1)
            fake_B_label = self.cyclegan.netD_B(fake_B).reshape(-1)
            BA_lossG = self.criterionG(fake_A_label, real_label)
            AB_lossG = self.criterionG(fake_B_label, real_label)

            # Cycle loss: L1 loss
            BA_lossC = self.criterionCycle(real_B, rec_B)*10.
            AB_lossC = self.criterionCycle(real_A, rec_A)*10.

            # Final loss
            lossG = BA_lossC + AB_lossC + BA_lossG + AB_lossG

            # Update Generator
            lossG.backward()
            self.optimizerG.step()

            ### Show training result
            if batch_idx % 100 == 0:
                print(('Epoch [{}:{}], Step [{}%] -- '
                    'lossD_A: {:.5f}, lossD_B: {:.5f}, '
                    'lossG_AB: {:.5f}, lossG_BA: {:.5f}').format(
                        epoch, self.config['train']['n_epochs'],
                        int(100*batch_idx/len(self.dataloader)),
                        A_lossD.item(), B_lossD.item(),
                        AB_lossG.item()+AB_lossC.item(),
                        BA_lossG.item()+BA_lossC.item()))

                self.cyclegan.eval()
                with torch.no_grad():
                    fake_B = self.cyclegan.netG_AB(self.fixed_real_A)
                    fake_A = self.cyclegan.netG_BA(self.fixed_real_B)
                    rec_B = self.cyclegan.netG_AB(fake_A)
                    rec_A = self.cyclegan.netG_BA(fake_B)

                    imgs_ABA = torch.cat([self.fixed_real_A, fake_B, rec_A])
                    imgs_BAB = torch.cat([self.fixed_real_B, fake_A, rec_B])
                    img_grid_ABA = torchvision.utils.make_grid(imgs_ABA, nrow=5, normalize=True)
                    img_grid_BAB = torchvision.utils.make_grid(imgs_BAB, nrow=5, normalize=True)

                    self.writer_ABA.add_image('Domain ABA', img_grid_ABA, batch_idx+epoch*len(self.dataloader))
                    self.writer_BAB.add_image('Domain BAB', img_grid_BAB, batch_idx+epoch*len(self.dataloader))

    def validate(self):
        pass

    def finalize(self):
        pass
