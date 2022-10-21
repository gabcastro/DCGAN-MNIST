from torch.utils.data import DataLoader as TorchLoader

class DataLoader:
    def __init__(self, dataset):
        self.batch_size = 128
        self.dataset = dataset
        self.trainloader = None
        self.dataiter = None

        self.generateloader()

    def generateloader(self):
        self.trainloader = TorchLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        print(f'Total of batchs in trainloader: {len(self.trainloader)}')
        self.dataiter = iter(self.trainloader)

    def getnext(self):
        return self.dataiter.next()