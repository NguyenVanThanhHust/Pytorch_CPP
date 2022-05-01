import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_no_obj = 0.5
        self.lambda_coord = 5
    
    def forward(self, preds, tgts):
        preds = preds.reshape(-1, self.S, self.S, self.C + self.B*5)
        iou_b1 = intersection_over_union(preds[..., self.C+1:self.C + (self.B-1) * 5], 
                                            tgts[..., self.C+1:self.C + (self.B-1) * 5], 
                                            box_format="xywh")
        iou_b2 = intersection_over_union(preds[..., self.C + (self.B-1) * 5+1:self.C + self.B * 5], 
                                            tgts[..., self.C+1:self.C + (self.B-1) * 5], 
                                            box_format="xywh")

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = tgts[..., self.C].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        box_preds = exists_box * (
            best_box * preds[..., self.C + (self.B-1) * 5+1:self.C + self.B * 5] +
            (1 - best_box) * preds[..., self.C+1:self.C + (self.B-1) * 5]
        )

        box_tgts = exists_box * tgts[..., self.C+1:self.C + (self.B-1) * 5]

        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(
            torch.abs(box_preds[..., 2:4] + 1e-6)
        )
        box_tgts[..., 2:4] = torch.sqrt(box_tgts[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_preds, end_dim=-2), 
            torch.flatten(box_tgts, end_dim=-2)
        )

        # ======================== #
        #   FOR OBJECT LOSS        #
        # ======================== #

        pred_box = (
            best_box * preds[..., self.C + (self.B-1) * 5:self.C + (self.B-1) * 5+1] + (1 - best_box) * preds[..., self.C:self.C+1]
        )
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box), 
            torch.flatten(exists_box * tgts[..., self.C:self.C+1])
        )

        # ======================== #
        #   FOR NO OBJECT LOSS     #
        # ======================== #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * preds[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * tgts[..., self.C:self.C+1], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * preds[..., self.C + (self.B-1) * 5:self.C + (self.B-1) * 5+1], start_dim=1),
            torch.flatten((1 - exists_box) * tgts[..., self.C:self.C+1], start_dim=1)
        )

        # ======================== #
        #   FOR CLASS LOSS         #
        # ======================== #

        class_loss = self.mse(
            torch.flatten(exists_box * preds[..., :self.C], end_dim=-2), 
            torch.flatten(exists_box * tgts[..., :self.C], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_no_obj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss