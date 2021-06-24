import numpy as np
from torch.utils.data import dataset
from torch.utils.data.dataset import T_co
from torchvision.transforms import transforms


class NumpyLoader(dataset.Dataset):
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.label[index]
        return self.trans(img), label, index

    def __init__(self, label_path, feature_path):
        self.imgs = np.load(feature_path)
        self.label = np.load(label_path)
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.label)

    def unique_class(self):
        print(np.unique(self.label))
        return len(np.unique(self.label))
