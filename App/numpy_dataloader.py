import PIL
import numpy as np
from torch.utils.data import dataset
import torch
import clip
from torchvision.transforms import transforms


class NumpyLoader(dataset.Dataset):
    def __getitem__(self, index):
        instance = self.instances[index]
        label = self.label[index]
        if self.pre_process == None:
            return instance, label, index
        return self.pre_process(instance), label, index
        # if self.pre_process is None:
        #     return img, label, index

        # # torch_tensor = transforms.ToPILImage(torch_tensor)
        # to_img = PIL.Image.fromarray(img)


        # self.pre_process(to_img)
        # transform = transforms.Compose([transforms.PILToTensor()])
        # tensor = transform(to_img)

        # return tensor, label, index

    def __init__(self, label_path, feature_path, pre_process):
        self.instances = np.load(feature_path)
        self.noise_or_not = None
        self.label = np.int8(np.load(label_path))
        

        if len(self.instances[0].shape) == 3 and self.instances[0].shape[2] == 3:
            self.pre_process = pre_process
        elif len(self.instances[0].shape) == 3 and self.instances[0].shape[2] == 1:
            self.instances = np.repeat(self.instances, 3, axis=2)
            self.pre_process = pre_process
        elif len(self.instances[0].shape) == 2:
            self.instances = np.expand_dims(self.instances, axis = 2)
            self.instances = np.repeat(self.instances, 3, axis=2)
            self.pre_process = pre_process
        elif isinstance(self.instances[0], str):
            self.instances = [clip.tokenize(i) for i in self.instances]
            self.pre_process = pre_process
        else:
            self.pre_process = None

    def __len__(self):
        return len(self.label)

    def unique_class(self):
        print(np.unique(self.label))
        return len(np.unique(self.label))
