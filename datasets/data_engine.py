from torchvision import transforms
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import glob
import os
from einops import rearrange
from tutils.mn.data.tsitk import read
from tqdm import tqdm
import monai
from monai.transforms import SpatialPadd, CenterSpatialCropd, Resized, NormalizeIntensityd
from monai.transforms import RandAdjustContrastd, RandShiftIntensityd, Rotated, RandAffined
from tutils import tfilename, tdir

from .dataset2d import Dataset2D
import numpy as np
import matplotlib.pyplot as plt
import cv2
from einops import repeat


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

class PointPromptGenerator(object):
    def __init__(self, size) -> None:
        self.size = size
        self.xy = np.arange(0, self.size[0] * self.size[1])
        assert size == (1024,1024), f"Got {size}"        

    def get_prompt_point(self, gt_mask):
        assert gt_mask.shape == (1024,1024), f"Got {gt_mask.shape}"
        gt_mask = np.float32(gt_mask>0)
        prob = rearrange(gt_mask, "h w -> (h w)")
        prob = prob / prob.sum()
        loc = np.random.choice(a=self.xy, size=1, replace=True, p=prob)[0]
        x, y = loc % self.size[1], loc // self.size[1]
        return x, y

    @staticmethod
    def select_random_subsequent_point(pred_mask, gt_mask):
        intersect_mask = np.logical_and(pred_mask, gt_mask)
        diff_mask = np.abs(pred_mask - gt_mask)
        error_mask = diff_mask - intersect_mask
        flat_error_mask = error_mask.reshape(-1)
        error_indices = np.where(flat_error_mask > 0)[0]
        if len(error_indices) == 0:
            return None  # No error pixels found
        error_rows, error_cols = np.unravel_index(error_indices, error_mask.shape)
        non_intersect_indices = np.where(~intersect_mask[error_rows, error_cols])[0]
        if len(non_intersect_indices) == 0:
            return None  # No error pixels found outside intersected regions
        selected_index = np.random.choice(non_intersect_indices)
        y, x = error_rows[selected_index], error_cols[selected_index]
        if pred_mask[y, x] == 1 and gt_mask[y, x] == 0:
            classification = 1
        else:
            classification = 0
        return (x, y), classification

    @staticmethod
    def select_random_subsequent_point_torch(predictions, ground_truths):
        b, c, h, w = predictions.size()
        device = predictions.device
        points = []
        assert predictions.shape == ground_truths.shape, f"Got { predictions.shape, ground_truths.shape}"
        for i in range(b):
            pred_mask = predictions[i]
            gt_mask = ground_truths[i]
            intersection = (pred_mask * gt_mask).view(-1)
            union = (pred_mask + gt_mask).view(-1)
            error_region = ((pred_mask != gt_mask) * (union != 0)).view(-1)
            error_region_indices = torch.where(error_region == 1)[0]
            if len(error_region_indices) == 0:
                # If there are no error regions, randomly sample a point from the whole image
                point_index = torch.randint(0, h * w, (1,))
            else:
                # Sample a point from the error region
                point_index = torch.randint_like(error_region_indices, 0, len(error_region_indices))
                point_index = error_region_indices[point_index]
            point = torch.zeros((h * w,), device=device)
            point[point_index] = 1
            point = point.view((1, 1, h, w))
            if (pred_mask[0] == gt_mask[0]).all():
                # If the mask predictions match the ground truth, the point is background
                point = 1 - point
            points.append(point)
        return torch.cat(points, dim=0)



class BoxPromptGenerator(object):
    def __init__(self, size) -> None:
        self.size = size

    @staticmethod
    def mask_to_bbox(mask):
        # Find the indices of all non-zero elements in the mask
        coords = np.nonzero(mask)

        # Compute the minimum and maximum values of the row and column indices
        x_min = np.min(coords[1])
        y_min = np.min(coords[0])
        x_max = np.max(coords[1])
        y_max = np.max(coords[0])

        # Return the coordinates of the bounding box
        return (x_min, y_min, x_max, y_max)
        # return (y_min, x_min, y_max, x_max)

    def add_random_noise_to_bbox(self, bbox):
        bbox = list(bbox)
        # Calculate the side lengths of the box in the x and y directions
        x_side_length = bbox[2] - bbox[0]
        y_side_length = bbox[3] - bbox[1]

        # Calculate the standard deviation of the noise
        std_dev = 0.1 * (x_side_length + y_side_length) / 2

        # Generate random noise for each coordinate
        x_noise = np.random.normal(scale=std_dev)
        y_noise = np.random.normal(scale=std_dev)

        # Add the random noise to each coordinate, but make sure it is not larger than 20 pixels
        bbox[0] += min(int(round(x_noise)), 20)
        bbox[1] += min(int(round(y_noise)), 20)
        bbox[2] += min(int(round(x_noise)), 20)
        bbox[3] += min(int(round(y_noise)), 20)

        # Make sure the modified coordinates do not exceed the maximum possible values
        bbox[0] = max(bbox[0], 0)
        bbox[1] = max(bbox[1], 0)
        bbox[2] = min(bbox[2], self.size[0])
        bbox[3] = min(bbox[3], self.size[1])

        # Return the modified bounding box
        return bbox

    def get_prompt_box(self, gt_mask):
        """ return (x_min, y_min, x_max, y_max) """
        assert gt_mask.shape == (1024,1024), f"Got {gt_mask.shape}"
        box = self.mask_to_bbox(gt_mask)
        box_w_noise = self.add_random_noise_to_bbox(box)
        return box_w_noise

# CACHE_DISK_DIR="/home1/quanquan/code/projects/medical-guangdong/cache/data2d_3/"

class DataEngine(Dataset):
    def __init__(self, dirpath=None, img_size=(1024,1024)) -> None:
        super().__init__()    
        self.expand_dataset_ratio = 2 # expand_dataset
        self.point_prompt_generator = PointPromptGenerator(img_size)
        self.box_prompt_generator = BoxPromptGenerator(img_size)
        self._get_dataset(dirpath=dirpath)
    
    def _get_dataset(self, dirpath):
        self.dataset = Dataset2D(dirpath=dirpath, is_train=True)    

    def __len__(self):
        return len(self.dataset) * self.expand_dataset_ratio

    def _get_true_index(self, idx):
        return idx % self.expand_dataset_ratio

    def __getitem__(self, idx):
        return self.get_prompt(idx)

    def get_prompt_point(self, gt_mask):
        return self.point_prompt_generator.get_prompt_point(gt_mask)

    def get_prompt_box(self, gt_mask):
        return self.box_prompt_generator.get_prompt_box(gt_mask)

    def get_prompt(self, idx):
        idx = self._get_true_index(idx)
        data = self.dataset.__getitem__(idx)

        data['img'] = repeat(data['img'], "1 h w -> b h w", b=3)
        gt_mask = data['label'][0].numpy()

        # if np.random.rand() > 0.5:
        prompt_point = self.get_prompt_point(gt_mask)
        # else:
        prompt_box =  self.get_prompt_box(gt_mask)

        data['prompt_point'] = torch.IntTensor(prompt_point)
        data['prompt_box'] = torch.IntTensor(prompt_box)
        return data
        
    def get_subsequent_prompt_point(self, pred_mask, gt_mask):
        # return self.point_prompt_generator.select_random_subsequent_point_torch(pred_mask, gt_mask)
        # return self.point_prompt_generator.select_random_subsequent_point(pred_mask=pred_mask, gt_mask=gt_mask)
        coord_collect = []
        label_collect = []
        for i in range(pred_mask.shape[0]):
            coords, label = self.point_prompt_generator.select_random_subsequent_point(pred_mask[i][0], gt_mask[i][0])
            coord_collect.append(coords)
            label_collect.append(label)
        return np.stack(coord_collect), np.stack(label_collect)


    # def get_prompt_mask(self, )

class ValidEngine(DataEngine):
    def __init__(self, dirpath=None, img_size=(1024,1024)) -> None:
        assert dirpath is not None
        super().__init__(dirpath=dirpath)
        self.expand_dataset_ratio = 1

    def _get_dataset(self, dirpath):
        self.dataset = Dataset2D(dirpath=dirpath, is_train=False)  

    def __len__(self):
        return len(self.dataset)

    def _get_true_index(self, idx):
        return idx


if __name__ == "__main__":
    dataset = DataEngine()
    data = dataset.__getitem__(0)

    import ipdb; ipdb.set_trace()
