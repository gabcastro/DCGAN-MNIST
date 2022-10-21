from dataloader import DataLoader
from dataset import Dataset
from discriminator import Discriminator

import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from torchsummary import summary

def show_tensor_images(tensor_img, num_imgs = 16, size=(1, 28, 28)):
    unflat_img = tensor_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_imgs], nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()

def main():
    
    device = 'cpu'
    noise_dim = 64

    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.99

    epochs = 20

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

if __name__ == "__main__":
    main()