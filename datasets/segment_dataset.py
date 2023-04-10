
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
import monai
from monai.transforms import SpatialPadd, CenterSpatialCropd, Resized, NormalizeIntensityd
from monai.transforms import RandAdjustContrastd, RandShiftIntensityd, Rotated, RandAffined

from tutils import tfilename


DEFAULT_PATH='/home1/quanquan/datasets/lsw/seg_data/type1/'

LABEL_INDICES={
            "t2sag": ["bg","kidney", "label 2", "label 3", "rectum", "tumor", "other"],
        }

DEFAULT_CONFIG={
    "pad": (512,512,100),
    "crop": (512,512,64),
    "resize": (128,128,64),
}

class TrainDatasetset(dataset):
    def __init__(self, is_train=True, dirpath=DEFAULT_PATH, mod_to_collect=None, transforms=None, config_dataset=DEFAULT_CONFIG, use_cache=True):
        """
            mod_to_collect: {
                "t2sag": ["kidney", ...]
            }
        """
        self.is_train = is_train
        self.dirpath = os.path.join(dirpath)
        self.labels_indices = LABEL_INDICES
        # self.mod_to_collect = self.check_modalities(mod_to_collect)
        img_paths = self.prepare_datalist()
        self.split_datasets(img_paths)
        self.cache_dirpath = '/home1/quanquan/code/projects/medical-guangdong/lsw_data/cached_segment_data/'
        self.use_cache = use_cache

        self.preprocess_transform = monai.transforms.Compose([
            SpatialPadd(keys=[f'img', f'label'], spatial_size=DEFAULT_CONFIG['pad']),
            CenterSpatialCropd(keys=[f'img', f'label'], roi_size=DEFAULT_CONFIG['crop']),
            Resized(keys=[f'img', f'label'], spatial_size=DEFAULT_CONFIG['resize']),
            NormalizeIntensityd(keys=[f'img'], subtrahend=-440, divisor=505),        
            ]
        )
        self.augtransforms = monai.transforms.Compose([
            # RandShiftIntensityd(keys=[f'img'], ),
            Rotated(keys=[f'img', f'label'], angle=10),
            RandAffined(keys=[f'img', f'label'], shear_range=0.1, translate_range=(5,5,5), scale_range=(0.95,1.1)),
            RandAdjustContrastd(keys=[f'img'], gamma=(0.5, 2.5), prob=0.5),
        ])

    def prepare_datalist(self):
        names = glob.glob(self.dirpath + "images/*")
        names = [name.split('/')[-1] for name in names]
        names.sort()
        assert len(names) > 0
        # print("Debug: ", names)
        # for mod in self.mod_to_collect:
        return names
    
    def split_datasets(self, img_paths):
        valset_len = int(len(img_paths) / 4)
        trainset_len = len(img_paths) - valset_len
        self.val_names = img_paths[:valset_len]
        self.train_names = img_paths[valset_len:]

    def check_modalities(self, mod_dict:dict) -> dict:
        assert mod_dict is not None
        available_mods = self.labels_indices.keys()
        ret_dict = {}
        if isinstance(mod_dict, dict):
            for mod, anas in mod_dict.items():
                assert mod in available_mods
                assert isinstance(anas, list)
                for ana in anas:
                    assert isinstance(ana, str)
            return mod_dict
        else:
            raise NotImplementedError

    def _get_data(self, index):
        img_names = self.train_names if self.is_train else self.val_names
        if not self.use_cache:
            ret_dict = {}
            img_path = os.path.join(self.dirpath, 'images' ,img_names[index])
            label_path = os.path.join(self.dirpath, 'ROI', img_names[index]+'.nrrd')
            # print(label_path)
            try:
                img = read(img_path, "dicom")
                img = sitk.GetArrayFromImage(img)
                label = read(label_path, "nrrd")
                label = sitk.GetArrayFromImage(label)
            except Exception as e:
                # print("Error message: ", e)
                print("Path: ", img_path, label_path)
                # raise ValueError
            img = rearrange(img, "d h w -> h w d").astype(float)
            label = rearrange(label, "d h w -> h w d").astype(float)
            # print("debug", img.shape, label.shape)
            ret_dict[f"img"] = img[None]
            ret_dict[f"label"] = label[None]
            ret_dict["path"] = img_path
            if self.augtransforms is not None:
                ret_dict = self.augtransforms(ret_dict)
            # print("dataset debug ", ret_dict[f"img"].shape)
            # for ana_name in anas:
            #     ana_index = self.labels_indices[mod].index(ana_name)
            #     ana = np.float32(ret_dict[f"label"] == ana_index)
            #     ret_dict[f"{ana_name}"] = ana
        else:
            ret_dict = np.load(os.path.join(self.cache_dirpath, img_names[index]+".npy"), allow_pickle=True).tolist()
        
        return ret_dict
    
    def _image_register(self, data):
        # TODO: image registration
        pass

    def __getitem__(self, index):
        data = self._get_data(index)
        return data

    def __len__(self):
        if self.is_train:
            return len(self.train_names)
        else:
            return len(self.val_names)
        
    def preprocess_before_training(self):
        self.use_cache = False
        # augtransforms = monai.transforms.Compose([
        #     SpatialPadd(keys=[f'img', f'label'], spatial_size=DEFAULT_CONFIG['pad']),
        #     CenterSpatialCropd(keys=[f'img', f'label'], roi_size=DEFAULT_CONFIG['crop']),
        #     Resized(keys=[f'img', f'label'], spatial_size=DEFAULT_CONFIG['resize']),
        #     NormalizeIntensityd(keys=[f'img'], subtrahend=-440, divisor=505),        
        #     ]
        # )
        self.is_train = True
        for i in tqdm(range(len(self.train_names))):
            filename = tfilename(self.cache_dirpath, self.train_names[i]+".npy")
            if not os.path.exists(filename):
                cached = self._get_data(i)
                np.save(filename, cached)
        self.is_train = False
        for i in tqdm(range(len(self.val_names))):
            filename = tfilename(self.cache_dirpath, self.val_names[i]+".npy")
            if not os.path.exists(filename):
                cached = self._get_data(i)
                np.save(filename, cached)
            # cached = self._get_data(i)
            # np.save(tfilename(self.cache_dirpath, self.train_names[i]+".npy"), cached)


class TestTrainset(TrainDatasetset):
    def __init__(self, dirpath=DEFAULT_PATH, mod_to_collect=None, transforms=None, config_dataset=DEFAULT_CONFIG):
        super().__init__(dirpath, mod_to_collect, transforms, config_dataset)
        # self.augtransforms = None
        self.augtransforms = {}
        for mod in mod_to_collect.keys():
            self.augtransforms[mod] = monai.transforms.Compose([
                # SpatialPadd(keys=[f'img', f'label'], spatial_size=DEFAULT_CONFIG['pad']),
                # CenterSpatialCropd(keys=[f'img', f'label'], roi_size=DEFAULT_CONFIG['crop']),
                Resized(keys=[f'img'], spatial_size=DEFAULT_CONFIG['resize']),
                NormalizeIntensityd(keys=[f'img'], subtrahend=-440, divisor=505),
                ]
            )


if __name__ == "__main__":
    dataset = TrainDatasetset(is_train=False)
    # dataset.preprocess_before_training()
    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)
        import ipdb; ipdb.set_trace()
    # data = dataset.__getitem__(0)