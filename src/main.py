from dataloader import DataLoader
from dataset import Dataset
from discriminator import Discriminator
from generator import Generator
from loss import *

import matplotlib.pyplot as plt
from tqdm import tqdm

from torch import nn, optim, randn
from torchvision.utils import make_grid
from torchsummary import summary

def show_tensor_images(tensor_img, num_imgs = 16, size=(1, 28, 28)):
    unflat_img = tensor_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_imgs], nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)

def main():
    
    device = 'cpu'
    noise_dim = 64

    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.99

    epochs = 1

    ds = Dataset()
    ds_mnist = ds.load_mnist()

    image, label = ds_mnist[4]
    # plt.imshow(image.squeeze(), cmap='gray')
    # plt.waitforbuttonpress()
    print(f'Total of images on trainset: {len(ds_mnist)}')

    dl = DataLoader(ds_mnist)
    images, _ = dl.getnext()
    print(images.shape)

    show_tensor_images(images)

    D = Discriminator()
    D.to(device)

    summary(D, input_size=(1, 28, 28))

    G = Generator(noise_dim)
    G.to(device)

    summary(G, input_size=(1, noise_dim))

    D = D.apply(weights_init)
    G = G.apply(weights_init)

    D_opt = optim.Adam(D.parameters(), lr = lr, betas = (beta_1, beta_2))
    G_opt = optim.Adam(G.parameters(), lr = lr, betas = (beta_1, beta_2))

    for i in range(epochs):

        total_d_loss = 0.0
        total_g_loss = 0.0

        for real_img, _ in tqdm(dl.trainloader):

            real_image = real_img.to(device)
            noise = randn(dl.batch_size, noise_dim, device=device)

            # find loss and update weights for D

            D_opt.zero_grad()

            fake_img = G(noise)
            D_pred = D(fake_img)
            D_fake_loss = fake_loss(D_pred)

            D_pred = D(real_image)
            D_real_loss = real_loss(D_pred)

            D_loss = (D_fake_loss + D_real_loss)/2

            total_d_loss += D_loss.item()

            D_loss.backward()
            D_opt.step()

            # find loss and update weights for G

            G_opt.zero_grad()

            noise = randn(dl.batch_size, noise_dim, device=device)

            fake_img = G(noise)
            D_pred = D(fake_img)
            G_loss = real_loss(D_pred)

            total_g_loss += G_loss.item()

            G_loss.backward()
            G_opt.step()

        avg_d_loss = total_d_loss / len(dl.trainloader)
        avg_g_loss = total_g_loss / len(dl.trainloader)

        print(f'Epoch {i+1} / D loss: {avg_d_loss} / G loss: {avg_g_loss}')

        show_tensor_images(fake_img)


if __name__ == "__main__":
    main()