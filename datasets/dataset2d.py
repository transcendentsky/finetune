from torchvision import transforms
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
import glob
import os
from einops import rearrange
from tutils.mn.data.tsitk import read
from tqdm import tqdm
# from monai.transforms import SpatialPadd, CenterSpatialCropd, Resized, NormalizeIntensityd
# from monai.transforms import RandAdjustContrastd, RandShiftIntensityd, Rotated, RandAffined
# from datasets.common_2d_aug import RandomRotation, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
from tutils import tfilename, tdir
import random

# DEFAULT_PATH='/home1/quanquan/datasets/KiTS/'
DEFAULT_PATH="/home1/quanquan/datasets/BCV-Abdomen/Training/"

LABEL_INDICES={
            "t2sag": ["bg","kidney", "label 2", "label 3", "rectum", "tumor", "other"],
        }

# CACHE_DISK_DIR="/home1/quanquan/code/projects/medical-guangdong/cache/data2d_3/"
CACHE_DISK_DIR=None
# DEFAULT_CONFIG={
#     "pad": (512,512),
#     "crop": (384,384),
#     "resize": (512,512),
# }

class Dataset2D(dataset):
    def __init__(self, dirpath=None, is_train=True) -> None:
        super().__init__()
        if is_train:
            self.dirpath = os.path.join(dirpath)
        else:
            self.dirpath = os.path.join(dirpath, 'test/')
        self.is_train = is_train
        self.img_names = self.prepare_datalist()
        self.prepare_transforms()

    def prepare_transforms(self):
        self.geo_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),  
            # transforms.ToTensor(),
            transforms.RandomCrop((448, 448)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.Resize((1024,1024)),
        ])
        # self.resize_1024 = 
        # self.resize_256 = transforms.Resize((256,256))
        self.int_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1]),
        ])
        self.to_tensor = transforms.ToTensor()

        self.test_transform_img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1]),
            transforms.Resize((1024,1024)),
        ])
        self.test_transform_label = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),  
            transforms.ToTensor(),
            transforms.Resize((1024,1024)),
        ])

    def __len__(self):
        return len(self.img_names)
    
    def prepare_datalist(self):
        names = glob.glob(self.dirpath + "*_img.npy")
        names = [os.path.split(name)[-1].replace("_img.npy", "") for name in names]
        names.sort()
        assert len(names) > 0, f"Got dirpath: {self.dirpath}"
        return names

    def _get_data(self, index):
        img_names = self.img_names
        img = np.load(os.path.join(self.dirpath, img_names[index]+"_img.npy"))
        label = np.load(os.path.join(self.dirpath, img_names[index]+"_label.npy"))

        # assert label.sum() > 100, f"Data Error!!! Got label sum() {label.sum()}"

        label_num = int(label.max())
        label_idx = np.random.choice(range(1, label_num+1))
        label = np.float32(label==label_idx)

        # import ipdb; ipdb.set_trace()
        img = np.clip(img, -200,400)
        if self.is_train:
            seed = torch.randint(2147483647, size=(1,)).item()
            random.seed(seed)
            img = self.geo_transform(img)
            img = self.int_transform(img)
            random.seed(seed)
            label = self.geo_transform(label)
        else:
            img = self.test_transform_img(img)
            label = self.test_transform_label(label)
            # img = self.resize_1024(img)
        img = self.to_tensor(img) if not isinstance(img, torch.Tensor) else img
        label = self.to_tensor(label) if not isinstance(label, torch.Tensor) else label
        
        if label.sum() <= 10:
            return self._get_data((index+1)%len(self))
        # assert label.sum() > 100, f"Transform Error!!! Got label sum() {label.sum()}"

        ret_dict = {
            "img": img, 
            "label": label,
            "name": img_names[index],
        }        
        return ret_dict
    
    def __getitem__(self, index):
        return self._get_data(index)


class TrainDatasetset(dataset):
    def __init__(self, is_train=True, dirpath=DEFAULT_PATH, transforms=None, use_disk_cache=False):
        """
            mod_to_collect: {
                "t2sag": ["kidney", ...]
            }
        """
        self.is_train = is_train
        self.dirpath = os.path.join(dirpath)
        self.labels_indices = LABEL_INDICES
        img_paths = self.prepare_datalist()
        self.split_datasets(img_paths)
        self.use_disk_cache = use_disk_cache
    
    def split_datasets(self, img_paths):
        valset_len = int(len(img_paths) / 5)
        trainset_len = len(img_paths) - valset_len
        self.val_names = img_paths[:valset_len]
        self.train_names = img_paths[valset_len:]

    def prepare_datalist(self):
        # KiTS_case_00209_1000.nii.gz
        # KiTS_case_00209.nii.gz
        names = glob.glob(self.dirpath + "img/*")
        names = [name.split('/')[-1] for name in names]
        names.sort()
        assert len(names) > 0
        names = [name.replace("img", "").replace(".nii.gz", "") for name in names]
        return names

    def _get_data(self, index):
        img_names = self.train_names if self.is_train else self.val_names
        if self.use_disk_cache:
            ret_dict = np.load(os.path.join(self.cache_dirpath, img_names[index]+".npy"), allow_pickle=True).tolist()
            return ret_dict
        
        ret_dict = {}
        name = img_names[index]
        img_path = os.path.join(self.dirpath, 'img' ,"img"+img_names[index]+'.nii.gz')
        label_path = os.path.join(self.dirpath, 'label', "label"+img_names[index]+'.nii.gz')
        # print(label_path)
        try:
            img = read(img_path, "nifti")
            img = sitk.GetArrayFromImage(img)
            label = read(label_path, "nifti")
            label = sitk.GetArrayFromImage(label)
        except Exception as e:
            print("Path: ", img_path, label_path)
            print("Error message: ", e)
            raise ValueError
        # img = rearrange(img, "d h w -> h w d").astype(float)
        # label = rearrange(label, "d h w -> h w d").astype(float)
        # print("debug", img.shape, label.shape)
        ret_dict[f"img"] = img[None]
        ret_dict[f"label"] = label[None]
        ret_dict["path"] = img_path
        ret_dict['name'] = name
        
        return ret_dict
    
    def __getitem__(self, index):
        if self.augtransforms is not None:
            ret_dict = self.augtransforms(ret_dict)
        data = self._get_data(index)
        return data

    def __len__(self):
        if self.is_train:
            return len(self.train_names)
        else:
            return len(self.val_names)
        
    def prepare_disk_cache(self):
        # self.is_train = True 
        cache_dir = tdir(CACHE_DISK_DIR)
        if not self.is_train:
            cache_dir = tdir(CACHE_DISK_DIR, "test")
        names = self.train_names if self.is_train else self.val_names

        count = 0
        for imgid in range(len(names)):
            data = self._get_data(imgid)
            img = data['img'][0]
            label = data['label'][0]
            # print(data.shape)
            name = data['name']
            length = img.shape[0]
            assert img.shape[1:] == (512,512), f"Got {img.shape}"
            assert label.shape[1:] == (512,512), f"Got {label.shape}"
            print("Processing image ", imgid, end="\r")
            for sid in range(length):
                img_save_path = os.path.join(cache_dir, name+f"_s{sid:04d}_img.npy")
                label_save_path = os.path.join(cache_dir, name+f"_s{sid:04d}_label.npy")
                label_slice = label[sid]
                if label_slice.sum() <= 100:
                    continue
                else:
                    img_slice = img[sid]
                    np.save(img_save_path, img_slice)
                    np.save(label_save_path, label_slice)
                    count += 1
                    # if count ==2 :
                    #     import ipdb; ipdb.set_trace()

        # print("Over!")
        # print()
        # raise NotImplementedError

        
if __name__ == "__main__":
    dataset = TrainDatasetset(is_train=False)
    dataset.prepare_disk_cache()

    # dataset = Dataset2D()
    # data = dataset.__getitem__(0)
    # print(data['label'].max())
    import ipdb; ipdb.set_trace()
                
