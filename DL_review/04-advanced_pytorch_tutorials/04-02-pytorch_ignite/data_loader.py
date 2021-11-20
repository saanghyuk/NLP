import torch

from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset):
    # flatten : (28, 28) -> 28*28 
    def __init__(self, data, labels, flatten=True):
        self.data = data
        self.labels = labels
        self.flatten = flatten

        super().__init__()

    def __len__(self):
        # how many samples?
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.flatten:
            # from (28, 28) to 784
            x = x.view(-1)
        
        # x = (784, )
        # y = (1, )
        return x, y


def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def get_loaders(config):
    x, y = load_mnist(is_train=True, flatten=False)

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    # (60000, 28, 28)
    # Shuffle to 60000 
    indices = torch.randperm(x.size(0))
    train_x, valid_x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    train_y, valid_y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    
    # Shuffle means whether shuffling in each epoch
    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=True),
        batch_size=config.batch_size,
        # for training set, necessary to shuffle
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_x, test_y = load_mnist(is_train=False, flatten=False)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader
