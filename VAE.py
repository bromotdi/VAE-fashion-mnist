"""
    VAE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, latent_dim, device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )

    def sample(self, sample_size, mu=None, logvar=None):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        if mu == None:
            mu = torch.zeros((sample_size, self.latent_dim)).to(self.device)
        if logvar == None:
            logvar = torch.zeros(
                (sample_size, self.latent_dim)).to(self.device)
        with torch.no_grad():
            x_sample = torch.rand(
                (sample_size, self.latent_dim)).to(self.device)
            x_reconstruct = self.decoder(self.upsample(
                x_sample).view(-1, 64, 7, 7)).to(self.device)
            return x_reconstruct

    def z_sample(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.rand_like(std).to(self.device)
        return mu + eps * std

    @staticmethod
    def loss(x, recon, mu, logvar):
        KL = -1/2 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        BCE = F.binary_cross_entropy(recon, x, reduction='sum')
        return BCE + KL

    def forward(self, x):
        x_latent = self.encoder(x).view(-1, 64*7*7)
        mu = self.mu(x_latent)
        logvar = self.logvar(x_latent)
        z = self.z_sample(mu=mu, logvar=logvar)
        return self.decoder(self.upsample(z).view(-1, 64, 7, 7)), mu, logvar


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         # encoder part
#         self.fc1 = nn.Linear(784, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc31 = nn.Linear(256, 2)
#         self.fc32 = nn.Linear(256, 2)
#         # decoder part
#         self.fc4 = nn.Linear(2, 256)
#         self.fc5 = nn.Linear(256, 512)
#         self.fc6 = nn.Linear(512, 784)
        
#     def encoder(self, x):
#         h = F.relu(self.fc1(x))
#         h = F.relu(self.fc2(h))
#         return self.fc31(h), self.fc32(h) # mu, log_var
        
#     def decoder(self, z):
#         h = F.relu(self.fc4(z))
#         h = F.relu(self.fc5(h))
#         return F.sigmoid(self.fc6(h)) 
    
#     def forward(self, x):
#         mu, log_var = self.encoder(x.view(-1, 784))
#         z = self.sampling(mu, log_var)
#         return self.decoder(z), mu, log_var
    
#     def sampling(self, mu, log_var):
#         std = torch.exp(0.5*log_var)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu)
    
