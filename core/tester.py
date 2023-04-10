import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from datasets.data_engine import DataEngine, ValidEngine
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
import torch
from tutils import Recorder
from rich.progress import track
from segment_anything import sam_model_registry, SamPredictor
from datasets.dataset2d import Dataset2D
from einops import rearrange


def compute_iou(pred_masks, gt_masks):
    # Calculate intersection
    intersection = torch.sum(torch.logical_and(pred_masks, gt_masks), dim=(2,3))

    # Calculate union
    union = torch.sum(torch.logical_or(pred_masks, gt_masks), dim=(2,3))

    # Calculate IoU
    iou = intersection / union

    return iou

# def compute_iou(pred_mask, gt_mask):
#     intersection = torch.logical_and(pred_mask, gt_mask)
#     intersection = reduce(intersection, "b c h w -> b c", reduction='sum')
#     union = torch.logical_or(pred_mask, gt_mask)
#     union = reduce(union, "b c h w -> b c", reduction='sum') + 1e-8
#     iou = intersection / union # if union > 0 else 0
#     return iou


class MyRecorder(Recorder):
    def record_batch(self, loss):
        assert type(loss) == dict, f"Got {loss}"
        # print("debug record", loss)
        if self.loss_keys is None:
            self.loss_list = []
            self.loss_keys = loss.keys()
        l_list = []
        for key, value in loss.items():
            if type(value) == torch.Tensor:
                values = value.detach().cpu().item()
                for v in values:
                    l_list.append(v)
            elif type(value) in [np.ndarray, np.float64, np.float32, int, float]:
                for v in values:
                    l_list.append(v)
            elif type(value) in [str, bool]:
                pass
            else:
                print("debug??? type Error? , got ", type(value))
                print("debug??? ", key, value)
                l_list.append(float(value))
        self.loss_list.append(l_list)


CACHE_DISK_DIR="/home1/quanquan/code/cache/data2d_3/"

class Tester:
    def __init__(self, config) -> None:
        self.config = config
        self.config_test = config['test']
        self.recorder = MyRecorder(reduction="mean")
        self.rank = "cuda"

    def test(self, model, epoch=0, rank='cuda', *args, **kwargs):
        # model.eval()
        # testset = Dataset2D()
        valid_engine = ValidEngine(CACHE_DISK_DIR)
        self.data_engine = valid_engine
        # dataloader = DataLoader(valid_engine, batch_size=self.config_test['batch_size'], num_workers=0, shuffle=False, drop_last=False)

        for i, data in track(enumerate(valid_engine)):           
        # for i, data in enumerate(valid_engine):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.detach().cpu().numpy()
            res = self.testing_step(model, data)
            # print(res)
            self.recorder.record(res)
            # if i >3: break
            del data
        d = self.recorder.cal_metrics()
        print(d)
        return d

    def testing_step(self, model, data):
        img = data['img'][None, ...]
        # img = rearrange(img, "c h w-> h w c")

        gt_mask = torch.Tensor(data['label']).cuda()[None,...]     
        prompt_point = data['prompt_point'] # shape: (b, 2)
        prompt_point = rearrange(prompt_point, "c-> 1 c")

        batch_size = prompt_point.shape[0]
        point_label = np.ones((batch_size,))
        prompt_box = data['prompt_box']
        prompt_box = rearrange(prompt_box, "c-> 1 c")

        assert gt_mask.shape[2:] == (1024,1024),f"Got {gt_mask.shape}" # (934, 1090, 3)
        assert prompt_point.shape[1:] == (2,), f"Got {prompt_point.shape}"
        assert point_label.shape == prompt_point.shape[:1], f"Got {point_label.shape}"
        assert prompt_box.shape[1:] == (4,), f"Got {prompt_box.shape}"

        img = torch.Tensor(img).cuda()
        model.set_torch_image(img, img.shape[2:])
        # Stage 1: use the 1st prompt, box or point
        # assert img.shape[1:] == (3,1024,1024),f"Got {img.shape}"

        iou_details = {}
        # 1
        pred_masks, scores, logits = model.predict(
            point_coords=prompt_point,
            point_labels=point_label,  
            multimask_output=True,   
            return_logits=False,       
        )

        iou = compute_iou(pred_masks, gt_mask)
        iou, _ = torch.max(iou, axis=1)
        iou_details['init_point'] = iou
        # 2
        pred_masks2, scores2, logits2 = model.predict(
            point_coords=None,
            point_labels=None,  
            box=prompt_box,
            multimask_output=True,  
            return_logits=False,              
        )
        iou = compute_iou(pred_masks2, gt_mask)
        iou, _ = torch.max(iou, axis=1)
        iou_details['init_box'] = iou        


        gt_mask_np = gt_mask.detach().cpu().numpy()
        for step in range(8):
            # n
            sub_points, sub_labels = [], []
            best_pred_masks = self.select_best_mask(pred_masks, gt_mask)            
            best_pred_masks_np = best_pred_masks.detach().cpu().numpy()
            
            # import ipdb; ipdb.set_trace()
            mask_input = logits[0, np.argmax(scores[0].detach().cpu().numpy()), :, :]  # Choose the model's best mask

            # sub_points, sub_labels = self.data_engine.get_subsequent_prompt_point(best_pred_masks_np, gt_mask_np)
            sub_points, sub_labels = self.data_engine.point_prompt_generator.select_random_subsequent_point(best_pred_masks_np[0][0], gt_mask_np[0][0])
            # sub_points = torch.Tensor(sub_points).to(prompt_point.device)
            # sub_labels = torch.Tensor(sub_labels).to(prompt_point.device)
            # sub_points = sub_points.unsqueeze(1)
            # sub_labels = sub_labels.unsqueeze(1)
            sub_points = np.array(sub_points)[None,...].astype(int)
            sub_labels = np.array(sub_labels)[None,...]
            prompt_point = np.concatenate([prompt_point, sub_points], axis=0)
            point_label = np.concatenate([point_label, sub_labels], axis=0)

            pred_masks2, scores, logits = model.predict(
                point_coords=sub_points,
                point_labels=sub_labels,  
                mask_input=mask_input[None,...],
                multimask_output=False, 
            )
            
            iou = compute_iou(pred_masks2, gt_mask)
            iou, _ = torch.max(iou, axis=1)
            iou_details[f'point_{step+2}'] = iou

        return iou_details
    
    @staticmethod
    def select_best_mask(predictions, ground_truth):
        # Move tensors to the same device (if not already on the same device)
        # if predictions.device != ground_truth.device:
        #     predictions = predictions.to(ground_truth.device)

        # Compute IoU between each prediction and ground truth
        intersection = torch.sum(predictions * ground_truth, dim=(2, 3))
        union = torch.sum(predictions + ground_truth, dim=(2, 3)) - intersection
        iou = intersection / (union + 1e-6)

        # Select the prediction with maximum IoU for each image in the batch
        best_indices = torch.argmax(iou, dim=1)
        best_masks = torch.gather(predictions, 1, best_indices.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, predictions.shape[2], predictions.shape[3]))

        return best_masks


# def test(logger, config, args):
#     from .tester import Tester
#     # CACHE_DISK_DIR="../cache/data2d_3/"
#     sam_checkpoint = "../segment-anything/sam_vit_h_4b8939.pth"
#     device = "cuda"
#     model_type = "default"

    
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)

#     our_pth = '/home1/quanquan/code/projects/medical-guangdong/runs/sam/ddp/ddp/ckpt_v/model_latest.pth'
#     state_dict = torch.load(our_pth)
#     sam.load_state_dict(state_dict)

#     learner = SamPredictor(sam_model=sam)
#     tester = Tester(config=config)
#     tester.test(learner)


def test(logger, config, args):
    from datasets.data_engine import ValidEngine
    from .trainer import SamLearner
    sam_checkpoint = "../segment-anything/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    
    data_engine = ValidEngine(dirpath=CACHE_DISK_DIR, img_size=(1024,1024))
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    our_pth = '/home1/quanquan/code/projects/medical-guangdong/runs/sam/ddp/ddp/ckpt_v/model_latest.pth'
    state_dict = torch.load(our_pth)
    sam.load_state_dict(state_dict)

    sam.to(device=device)
    learner = SamLearner(sam_model=sam, config=config, data_engine=data_engine)
    tester = Tester(config=config)
    tester.test(learner)


if __name__ == "__main__":
    from tutils import TConfig
    from tutils import print_dict
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/config.yaml")
    parser.add_argument("--func", default="test")

    tconfig = TConfig(mode='sc', file=__file__, parser=parser)
    config = tconfig.get_config()
    args = tconfig.get_args()
    logger = tconfig.get_logger()

    # print_dict(config)
    eval(args.func)(logger, config, args)