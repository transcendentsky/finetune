import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
from PIL import Image
import torch
from torch import Tensor


class Normalize(transforms.Normalize):
    def __init__(self, keys, *args, **kwargs):
        self.keys = keys
        super().__init__(*args, **kwargs)

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        for k, img in imgs.items():
            if k in self.keys:
                imgs[k] = F.normalize(img, self.mean, self.std, self.inplace)
        return imgs


class ToTensor:
    def __init__(self, keys, *args, **kwargs):
        self.keys = keys
        super().__init__(*args, **kwargs)

    def __call__(self, imgs):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        for k, img in imgs.items():
            if k in self.keys:
                imgs[k] = F.to_tensor(img)
        return imgs


class ColorJitter(transforms.ColorJitter):
    def __init__(self, keys, *args, **kwargs):
        self.keys = keys
        super().__init__(*args, **kwargs)

    def forward(self, imgs):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                for k, img in imgs.items():
                    if k in self.keys:
                        imgs[k] = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                for k, img in imgs.items():
                    if k in self.keys:
                        imgs[k] = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                for k, img in imgs.items():
                    if k in self.keys:
                        imgs[k] = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                for k, img in imgs.items():
                    if k in self.keys:
                        imgs[k] = F.adjust_hue(img, hue_factor)
        return imgs


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, keys, *args, **kwargs):
        self.keys = keys
        super().__init__(*args, **kwargs)

    def forward(self, imgs):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            for k, img in imgs.items():
                if k in self.keys:
                    imgs[k] = F.hflip(img)
            return imgs
        return imgs


class RandomRotation(transforms.RandomRotation):
    def __init__(self, keys, *args, **kwargs):
        assert isinstance(keys, list), f"{keys}"
        self.keys = keys
        super().__init__(*args, **kwargs)

    def forward(self, imgs):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        if isinstance(imgs[self.keys[0]], Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(imgs[self.keys[0]])
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        for k, img in imgs.items():
            if k in self.keys:
                imgs[k] = F.rotate(img, angle, self.resample, self.expand, self.center, fill)
        return imgs


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, keys, *args, **kwargs):
        self.keys = keys
        super().__init__(*args, **kwargs)

    def forward(self, imgs):
        """
        Args:
            imgs (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(imgs[self.keys[0]], self.scale, self.ratio)
        for k, img in imgs.items():
            if k in self.keys:
                imgs[k] = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        return imgs


class RandomAffine(transforms.RandomAffine):
    def __init__(self, keys, *args, **kwargs) -> None:
        self.keys = keys
        super().__init__(*args, **kwargs)

    def forward(self, imgs):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        if isinstance(imgs[self.keys[0]], Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(imgs[self.keys[0]])
            else:
                fill = [float(f) for f in fill]

        img_size = F.get_image_size(imgs[self.keys[0]])

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        for k, img in imgs.items():
            if k in self.keys:
                imgs[k] = F.affine(img, *ret, interpolation=self.interpolation, fill=fill)
        return imgs


def usage():
    from torchvision.utils import save_image
    aug = transforms.Compose([
        RandomRotation(keys=['image', 'label'], degrees=(-10,10)),
        RandomResizedCrop(keys=['image', 'label'], size=(224,224)),
        RandomHorizontalFlip(keys=['image', 'label']),
        ColorJitter(keys=['image',], brightness=0.15, contrast=0.25),
        ToTensor(keys=['image',]),
        Normalize(keys=['image',], mean=[0], std=[1]),
    ])

    pth_img = '/home1/quanquan/code/mytools/tutils/tutils/_paper_writing/corgi1.jpg'
    img1 = Image.open(pth_img).convert('RGB')
    # img2 = Image.open(pth_img).convert('RGB')
    img2 = torch.zeros((19, 224, 224))
    print(img2.shape)

    imgs = aug({"image":img1, "label":img2})
    img1, img2 = imgs['image'], imgs['label']
    import ipdb; ipdb.set_trace()
    save_image(img1, "_tmp_img1.png")
    save_image(img2, "_tmp_img2.png")
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    usage()



