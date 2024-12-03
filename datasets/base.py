from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
from pathlib import Path
import numpy as np


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False, train=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.train = train
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0
        if self.train:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(self.image_size)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda img: F.crop(img, top=240, left=192, height=300, width=275)),
                transforms.RandomCrop(self.image_size),
            ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = np.load(img_path).astype(np.float32) # HWC -> 이미 [-1, 1]로 norm 되어 있음.
        except BaseException as e:
            print(img_path)
        
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)

        image = transform(image) # CHW

        image_name = Path(img_path).stem
        return image, image_name

