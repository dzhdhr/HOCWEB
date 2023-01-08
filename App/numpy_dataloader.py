import PIL
import numpy as np
from torch.utils.data import dataset
import torch
from torchvision.transforms import transforms


class NumpyLoader(dataset.Dataset):
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.label[index]
        if self.pre_process is None:
            return img, label, index

        # torch_tensor = transforms.ToPILImage(torch_tensor)
        to_img = PIL.Image.fromarray(img)


        self.pre_process(to_img)
        transform = transforms.Compose([transforms.PILToTensor()])
        tensor = transform(to_img)

        return tensor, label, index

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
