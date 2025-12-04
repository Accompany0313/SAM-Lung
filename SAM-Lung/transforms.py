import numpy as np
import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ComposeEval(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(target, np.ndarray):
            target = Image.fromarray(target)

        image = F.resize(image, size)

        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class Resize(object):
    def __init__(self, size=1024):

        if isinstance(size, int):
            self.size = (size, size)  
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            self.size = (int(size[0]), int(size[1]))
        else:
            raise ValueError(f"Size must be int or (height, width) tuple, got {type(size)}")

    def __call__(self, image, target):
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(target, np.ndarray):
            target = Image.fromarray(target.astype(np.uint8))

        image = F.resize(
            img=image,
            size=self.size,  
            interpolation=Image.BICUBIC
        )

        target = F.resize(
            img=target,
            size=self.size,
            interpolation=Image.NEAREST
        )

        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(target, np.ndarray):
            target = Image.fromarray(target)

        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(target, np.ndarray):
            target = Image.fromarray(target)

        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target):
       
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(target, np.ndarray):
            target = Image.fromarray(target)

        angle = random.choice(self.degrees) 
        image = F.rotate(image, angle)
        target = F.rotate(target, angle)
        return image, target

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        
        transform = T.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue
        )
        return transform(image)

class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(target, np.ndarray):
            target = Image.fromarray(target)

        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
       
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(target, np.ndarray):
            target = Image.fromarray(target)

        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class ToTensor(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        return image


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image
