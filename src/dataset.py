from torchvision import datasets, transforms as T

class Dataset:
    def __init__(self):
        self.train_augmentation = T.Compose([
            T.RandomRotation((-20, +20)),
            T.ToTensor() # (h, w, c) => (c, h, w)
        ])

    def load_mnist(self):
        mnist_train = datasets.MNIST(
            'MNIST/', 
            download=True,
            train=True,
            transform=self.train_augmentation
        )

        return mnist_train