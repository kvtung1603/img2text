from torchvision import transforms


class ImgAugTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 0.1)),
        ])

    def __call__(self, img):
        img = self.transform(img)
        img = transforms.ToPILImage()(img)
        return img