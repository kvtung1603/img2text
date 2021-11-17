import numpy as np
from PIL import Image
from imgaug import augmenters as iaa


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential()


    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        img = Image.fromarray(img)
        return img

