import torch
import torch.nn as nn
from .generator import ResnetGenerator
from .discriminator import NLayerDiscriminator
from .loss import LeastSquaresGenerativeAdversarialLoss as GANLoss
from .transform import Translation as apply_transform

class CycleGAN(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, ndf=64):
        """
        CycleGAN Model consisting of Generator, NLayerDiscriminator, and Loss functions.
        """
        super(CycleGAN, self).__init__()

        # Define Generators
        self.G_A2B = ResnetGenerator(input_nc, output_nc, ngf)  # Translate A → B
        self.G_B2A = ResnetGenerator(output_nc, input_nc, ngf)  # Translate B → A

        # Define Discriminators
        self.D_A = NLayerDiscriminator(input_nc, ndf)  # Discriminate A
        self.D_B = NLayerDiscriminator(output_nc, ndf)  # Discriminate B

        # Define Loss
        self.criterion_GAN = GANLoss()

    def forward(self, real_A, real_B):
        """ Forward pass of CycleGAN """

        fake_B = self.G_A2B(real_A)  # Translate A → B
        fake_A = self.G_B2A(real_B)  # Translate B → A
        return fake_A, fake_B

    def train_step(self, real_A, real_B, optimizer_G, optimizer_D):
        """ Single training step for CycleGAN """
        # print(f"Input shape before fixing: {real_B.shape}")
       
        fake_B = self.G_A2B(real_A)
        fake_A = self.G_B2A(real_B)

        # Adversarial loss
        loss_G_A2B = self.criterion_GAN(self.D_B(fake_B), True)
        loss_G_B2A = self.criterion_GAN(self.D_A(fake_A), True)

        loss_G = loss_G_A2B + loss_G_B2A

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        return loss_G.item()

    def generate(self, real_A):
        """ Generate synthetic images """
        print(f"Input shape before fixing: {real_A.shape}")

        if real_A.ndim == 3:
            real_A = real_A.unsqueeze(0)  # Ensure [1, C, H, W]

        print(f"Final shape before passing to generator: {real_A.shape}")

        assert real_A.ndim == 4, f"Expected 4D input, but got {real_A.shape}"
        
        with torch.no_grad():
            fake_B = self.G_A2B(real_A)
    
        return fake_B.squeeze(0)

