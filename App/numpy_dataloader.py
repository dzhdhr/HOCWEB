import numpy as np
from torch.utils.data import dataset


class NumpyLoader(dataset.Dataset):
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.label[index]
        if self.pre_process == None:
            return img, label, index
        return self.pre_process(img), label, index

    def __init__(self, label_path, feature_path, pre_process):
        self.imgs = np.load(feature_path)
        self.noise_or_not = None
        self.label = np.int8(np.load(label_path))
        self.pre_process = pre_process

    def __len__(self):
        return len(self.label)

    def unique_class(self):
        print(np.unique(self.label))
        return len(np.unique(self.label))
