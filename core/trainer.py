import torch
import torchvision
import numpy as np
from tutils.trainer import Trainer, LearnerModule
from torch.utils.data import DataLoader

from segment_anything.modeling import Sam
# from segment_anything.utils.transforms import ResizeLongestSide
from utils.transforms import ResizeLongestSide
from typing import Optional, Tuple

from .loss import compute_all_loss, ranked_combined_loss, compute_iou
from datasets.data_engine import DataEngine
from models.build_sam import sam_model_registry
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from einops import rearrange
from torch.nn import functional as F
# from einops import repeat

# class MyTrainer(Trainer):
#     def __init__(self, logger=None, config=None, tester=None, monitor=None, **kwargs):
#         super().__init__(logger, config, tester, monitor, **kwargs)    

def lr_schedule(epoch):
    if epoch < 250:
        return (epoch + 1) / 250 * 0.1
    elif epoch < 500:
        return 0.01
    else:
        return 0.001
           

class SamLearner(LearnerModule):
    def __init__(
        self,
        sam_model: Sam,
        config=None, 
        logger=None, 
        data_engine=None, 
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.config = config
        self.logger = logger
        self.model = sam_model
        self.net = self.model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()
        self.data_engine = data_engine
        self.features = None

    def save(self, pth, *args, **kwargs):
        # Default: "/model_epoch_{}.pth".format(epoch)
        torch.save(self.net.state_dict(), pth)
        return True

    def configure_optimizers(self, **kwargs):
        optimizer = optim.AdamW(params=self.model.parameters(), \
                           lr=self.config['training']['lr'], betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=self.config['training']['weight_decay'])
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule)
        return {'optimizer': optimizer, "scheduler": scheduler}

    def training_step(self, data, batch_idx, **kwargs):
        img = data['img']
        gt_mask = data['label']        
        prompt_point = data['prompt_point'] # shape: (b, 2)
        batch_size = prompt_point.shape[0]
        point_label = torch.ones((batch_size, 1))
        prompt_box = data['prompt_box']

        # print(img.shape)
        # print(prompt_point.shape, point_label.shape, prompt_box.shape)

        # Stage 1: use the 1st prompt, box or point
        # self.set_image(img)

        prompt_point = rearrange(prompt_point, "b c -> b 1 c")
        prompt_box = rearrange(prompt_box, "b c -> b 1 c")
        assert img.shape[1:] == (3,1024,1024),f"Got {img.shape}"
        assert prompt_point.shape[1:] == (1,2), f"Got {prompt_point.shape}"
        assert point_label.shape[1:] == (1,), f"Got {point_label.shape}"
        assert prompt_box.shape[1:] == (1,4), f"Got {prompt_box.shape}"

        self.set_torch_image(img, img.shape[2:])
        if np.random.random() > 0.5:
            pred_masks, iou_predictions, logits = self.predict_torch(
                point_coords=prompt_point,
                point_labels=point_label,  
                multimask_output=True,   
                return_logits=True,       
            )
        else:            
            pred_masks, iou_predictions, logits = self.predict_torch(
                point_coords=None,
                point_labels=None,  
                boxes=prompt_box,
                multimask_output=True,  
                return_logits=True,              
            )
        loss_1, _, _ = ranked_combined_loss(pred_mask=pred_masks, gt_mask=gt_mask, iou_pred=iou_predictions)

        # Stage 2: based on the above, add more points as prompts
        loss_details = {"loss_s1": loss_1}
        total_loss = loss_1

        gt_mask_small = F.interpolate(gt_mask, (256,256), mode="bilinear", align_corners=False,)
        gt_mask_small_np = gt_mask_small.detach().cpu().numpy()
        for step in range(8):
            sub_points, sub_labels = [], []
            best_pred_masks = self.select_best_mask(logits, gt_mask_small)
            
            best_pred_masks_np = best_pred_masks.detach().cpu().numpy()
            best_pred_masks_np = np.float32(best_pred_masks_np>0.25)
            sub_points_small, sub_labels_small = self.data_engine.get_subsequent_prompt_point(best_pred_masks_np, gt_mask_small_np)
            sub_points = torch.Tensor(sub_points_small).to(prompt_point.device) * 4  # from imgsize 256 -> imgsize 1024
            sub_labels = torch.Tensor(sub_labels_small).to(prompt_point.device) * 4
            sub_points = sub_points.unsqueeze(1)
            sub_labels = sub_labels.unsqueeze(1)

            # print("main Step ", step)
            _, scores, logits = self.predict_torch(
                point_coords=sub_points,
                point_labels=sub_labels,  
                mask_input=best_pred_masks,
                multimask_output=False, 
            )

            loss_2, _, _ = compute_all_loss(pred_mask=logits, gt_mask=gt_mask_small, iou_pred=scores)
            loss_details[f'loss_s2_{step}'] = loss_2
            # import ipdb; ipdb.set_trace()
            total_loss += loss_2

        return {"loss": total_loss, **loss_details}
    
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



    # def nnn():
    #     if point_coords is not None:
    #         assert (
    #             point_labels is not None
    #         ), "point_labels must be supplied if point_coords is supplied."
    #         point_coords = self.transform.apply_coords(point_coords, self.original_size)
    #         coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
    #         labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
    #         coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    #     if box is not None:
    #         box = self.transform.apply_boxes(box, self.original_size)
    #         box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
    #         box_torch = box_torch[None, :]
    #     if mask_input is not None:
    #         mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
    #         mask_input_torch = mask_input_torch[None, :, :, :]


    # ===============================================
    def predict_multi_prompt(
        self, 
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        mask_logits: Optional[torch.Tensor],  
    ):
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        
        if point_coords is not None:
            points = (coords_torch, point_labels)
        else:
            points = None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=None,
            masks=mask_logits,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        return masks, iou_predictions, low_res_masks

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        # masks = masks[0].detach().cpu().numpy()
        # iou_predictions = iou_predictions[0].detach().cpu().numpy()
        # low_res_masks = low_res_masks[0].detach().cpu().numpy()
        # return masks, iou_predictions, low_res_masks
        return masks, iou_predictions, low_res_masks

    # @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        sparse_embeddings, dense_embeddings = self._get_prompt_embedding(points, boxes, mask_input)

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        # import ipdb; ipdb.set_trace()

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks
    
    @torch.no_grad()
    def _get_prompt_embedding(self, points, boxes, mask_input):
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        return sparse_embeddings, dense_embeddings


    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
        

CACHE_DISK_DIR="/home1/quanquan/code/cache/data2d_3/"   

def train(logger, config, args):
    sam_checkpoint = "/home1/quanquan/code/projects/medical-guangdong/segment-anything/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"

    data_engine = DataEngine(dirpath=CACHE_DISK_DIR, img_size=(1024,1024))
    trainer = Trainer(logger, config)
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    learner = SamLearner(sam_model=sam, config=config, data_engine=data_engine)
    trainer.fit(learner, data_engine)


def test(logger, config, args):
    from .tester import Tester
    from datasets.data_engine import ValidEngine
    sam_checkpoint = "../segment-anything/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    
    data_engine = ValidEngine(dirpath=CACHE_DISK_DIR, img_size=(1024,1024))
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
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
    parser.add_argument("--func", default="train")

    tconfig = TConfig(mode='sc', file=__file__, parser=parser)
    config = tconfig.get_config()
    args = tconfig.get_args()
    logger = tconfig.get_logger()
    # print_dict(config)

    eval(args.func)(logger, config, args)