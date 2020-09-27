import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_features, out_features, kernel_size=3, stride=1),
                nn.InstanceNorm2d(out_features)
                )

    def forward(self, x):
        residual = x
        x = self.block(x)
        assert x.shape == residual.shape
        return residual + x


class Generator(nn.Module):
    """Resnet generator with 9 residual blocks"""
    def __init__(self, channels_img, features, n_blocks):
        super().__init__()
        self.channels_img = channels_img
        self.features = features
        self.n_blocks = n_blocks

        ### Encoder
        self.encoder = nn.Sequential(
                # (batch_size x channels_img x 256 x 256)
                nn.ReflectionPad2d(3),
                nn.Conv2d(channels_img, features, kernel_size=7, stride=1),
                nn.InstanceNorm2d(features),
                nn.ReLU(inplace=True),
                # (batch_size x features x 256 x 256)
                nn.Conv2d(features, features*2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(features*2),
                nn.ReLU(inplace=True),
                # (batch_size x features*2 x 128 x 128)
                nn.Conv2d(features*2, features*4, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(features*4),
                nn.ReLU(inplace=True),
                # (batch_size x features*4 x 64 x 64)
                )

        ### Transformer
        self.transform = nn.Sequential(*[ ResidualBlock(features*4, features*4)
                                        for _ in range(n_blocks) ])

        ### Decoder
        self.decoder = nn.Sequential(
                # (batch_size x features*4 x 64 x 64)
                nn.ConvTranspose2d(features*4, features*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(features*2),
                nn.ReLU(inplace=True),
                # (batch_size x features*2 x 128 x 128)
                nn.ConvTranspose2d(features*2, features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(features),
                nn.ReLU(inplace=True),
                # (batch_size x features x 256 x 256)
                nn.ReflectionPad2d(3),
                nn.Conv2d(features, channels_img, kernel_size=7, stride=1),
                nn.Tanh(),
                # (batch_size x channels_img x 256 x 256)
                )

    def forward(self, x):
        assert tuple(x.shape[1:]) == (self.channels_img, 256, 256)
        x = self.encoder(x)

        assert tuple(x.shape[1:]) == (self.features*4, 64, 64)
        x = self.transform(x)

        assert tuple(x.shape[1:]) == (self.features*4, 64, 64)
        x = self.decoder(x)

        assert tuple(x.shape[1:]) == (self.channels_img, 256, 256)
        return x


class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    def __init__(self, channels_img, features):
        super().__init__()
        self.channels_img = channels_img
        self.features = features

        self.net = nn.Sequential(
                # (batch_size x channels_img x 256 x 256)
                nn.Conv2d(channels_img, features, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # (batch_size x features x 128 x 128)
                nn.Conv2d(features, features*2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(features*2),
                nn.LeakyReLU(0.2, inplace=True),
                # (batch_size x features*2 x 64 x 64)
                nn.Conv2d(features*2, features*4, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(features*4),
                nn.LeakyReLU(0.2, inplace=True),
                # (batch_size x features*4 x 63 x 63)
                nn.Conv2d(features*4, features*8, kernel_size=4, stride=1, padding=1),
                nn.InstanceNorm2d(features*8),
                nn.LeakyReLU(0.2, inplace=True),
                # (batch_size x features*16 x  x 8)
                nn.Conv2d(features*8, 1, kernel_size=4, stride=1, padding=1)
                # (batch_size x 1 x 1 x 1)
                )

    def forward(self, x):
        assert tuple(x.shape[1:]) == (self.channels_img, 256, 256)
        x = self.net(x)
        return x


class ImagePool:

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class CycleGAN:
    """Wrapper class for components in CycleGAN

    Attributes
        netG_AB: generator model generating data from domain A to domain B
        netG_BA: generator model generating data from domain A to domain B
        netD_A: discriminator model for data in domain A
        netD_B: discriminator model for data in domain B
    """
    def __init__(self, channels_img, features_g, blocks_g, features_d, pool_size):
        self.netG_AB = Generator(channels_img=channels_img,
                                features=features_g,
                                n_blocks=blocks_g)
        self.netG_BA = Generator(channels_img=channels_img,
                                features=features_g,
                                n_blocks=blocks_g)
        self.netD_A = Discriminator(channels_img=channels_img,
                                features=features_d)
        self.netD_B = Discriminator(channels_img=channels_img,
                                features=features_d)

        ### Initialize models' weights
        self.netG_AB.apply(self._init_weight)
        self.netG_AB.apply(self._init_weight)
        self.netD_A.apply(self._init_weight)
        self.netD_B.apply(self._init_weight)

        ### Image Pools for generated images
        self.image_pool_A = ImagePool(pool_size)
        self.image_pool_B = ImagePool(pool_size)

    def to(self, device):
        self.netG_AB = self.netG_AB.to(device)
        self.netG_BA = self.netG_BA.to(device)
        self.netD_A = self.netD_A.to(device)
        self.netD_B = self.netD_B.to(device)
        return self

    def train(self):
        self.netG_AB.train()
        self.netG_BA.train()
        self.netD_A.train()
        self.netD_B.train()
        return self

    def eval(self):
        self.netG_AB.eval()
        self.netG_BA.eval()
        self.netD_A.eval()
        self.netD_B.eval()
        return self

    def forward(self):
        raise RuntimeError("You should run the components in model directly")

    def _init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight.data, 0.0, 0.2)
            init.zeros_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.normal_(m.weight.data, 0.0, 0.2)
            init.zeros_(m.bias.data)


if __name__ == "__main__":
    # Hyperparameters
    channels_img = 3
    input_size = 256
    features_g = 64
    blocks_g = 9
    features_d = 64
    pool_size = 50

    # Instantiate model
    cyclegan = CycleGAN(channels_img=channels_img,
                        features_g=features_g,
                        blocks_g=blocks_g,
                        features_d=features_d,
                        pool_size=pool_size)

    # Input data
    A_x = torch.rand(1, 3, 256, 256)
    B_x = torch.rand(1, 3, 256, 256)

    # Forward from A to B
    B_y = cyclegan.netG_AB(A_x)
    A_x_rev = cyclegan.netG_BA(B_y)

    # Forward from B to A
    A_y = cyclegan.netG_BA(B_x)
    B_x_rev = cyclegan.netG_AB(A_y)

    # Discirminator A
    cyclegan.netD_B(A_x)
    cyclegan.netD_B(A_x_rev)

    # Discriminator B
    cyclegan.netD_B(B_x)
    cyclegan.netD_B(B_x_rev)
