import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "train" if train else "val"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "mask")) if i.endswith(".png")]
        self.img_list = [os.path.join(data_root, "mask", i) for i in img_names]
        self.label_list = [os.path.join(data_root, "label", i) for i in img_names]

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        label = Image.open(self.label_list[idx]).convert('L')

        label = np.array(label) / 255 

        if self.transforms is not None:
            img, label = self.transforms(img, label)

        return img, label

    def __len__(self):
        return len(self.img_list)


    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs



