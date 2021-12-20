import numpy as np
from PIL import Image
from torch.utils.data import dataset
from torch.utils.data.dataset import T_co
from torchvision.transforms import transforms, Resize, CenterCrop


class NumpyLoader(dataset.Dataset):
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.label[index]
        return self.pre_process(img), label, index

    def __init__(self, label_path, feature_path,pre_process):
        self.imgs = np.load(feature_path)
        self.noise_or_not = None
        self.label = np.int8(np.load(label_path))
        print(self.imgs.shape)
        print(self.label.shape)
        self.pre_process = pre_process
        # self.imgs = self.pre_process(self.imgs)
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            # Resize(224, interpolation=Image.BICUBIC),
            # CenterCrop(224),
            # lambda image: image.convert("RGB"),
            # transforms.ToTensor(),
            # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            # # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.label)

    def unique_class(self):
        print(np.unique(self.label))
        return len(np.unique(self.label))
