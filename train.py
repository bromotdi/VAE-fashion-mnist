"""
    Training procedure for VAE
"""

import os
import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from VAE import Model

filepath = os.path.dirname(os.path.abspath(__file__))


def train(vae, trainloader, optimizer, device):
    vae.train()
    batch_loss = 0
    for n, (features, _) in enumerate(trainloader):
        features = features.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(features)
        loss = vae.loss(features, x_recon, mu=mu, logvar=logvar)
        loss.backward()
        batch_loss += loss.item()
        optimizer.step()
    return batch_loss / n


def test(vae, testloader, filename, epoch, device):
    vae.eval()  # set to inference mode
    with torch.no_grad():
        samples = vae.sample(64).to(device)
        a, b = samples.min(), samples.max()
        samples = (samples - a) / (b - a + 1e-10)
        torchvision.utils.save_image(torchvision.utils.make_grid(samples), filepath +
                                     '/samples/' + filename + 'epoch%d.png' % epoch)
        batch_loss = 0
        for n, (features, _) in enumerate(testloader):
            features = features.to(device)
            x_recon, mu, logvar = vae(features)
            loss = vae.loss(features, x_recon, mu=mu, logvar=logvar)
            batch_loss += loss.item()
    return batch_loss / n

def dequantization(x):
    return x + torch.zeros_like(x).uniform_(0., 1./256.)

def main(args):
    #  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu" # for mac
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(dequantization), #dequantization
        torchvision.transforms.Normalize((0.,), (257. / 256.,)),  # rescales to [0,1]
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset

    vae = Model(latent_dim=args.latent_dim, device=device).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    elbo_train = []
    elbo_test = []
    for epoch in tqdm(range(args.epochs)):
        epoch_train_loss = train(vae=vae, trainloader=trainloader, optimizer=optimizer, device=device)
        epoch_test_loss = test(vae=vae, testloader=testloader, filename=filename, epoch=epoch, device=device)
        elbo_train.append(epoch_train_loss)
        elbo_test.append(epoch_test_loss)
        print(f"Epoch {epoch + 1} finished:  train loss: {elbo_train[epoch]}, test loss: {elbo_test[epoch]}")
        if epoch % 10 == 0:
            torch.save(vae.state_dict(), filepath + "/models/" + filename + ".pkl")

    colors = {'train': 'blue', 'test': 'green'}
    linestyles = {'train': '-', 'test': '--'}
    fig, ax = plt.subplots()
    ax.plot(elbo_train, label=f'Train Loss', color=colors['train'], linestyle=linestyles['train'])
    ax.plot(elbo_test, label=f'Test Loss', color=colors['test'], linestyle=linestyles['test'])
    ax.set_title(f'Dataset: {args.dataset}')  
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.legend()
    ax.grid(True)
    plt.savefig(filepath + "/loss/" + f"{args.dataset}_loss.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='fashion-mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=50)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)
    args = parser.parse_args()
    main(args)
