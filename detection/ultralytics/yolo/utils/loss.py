# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import bbox_iou
from .tal import bbox2dist


def wasserstein_loss(pred, target, eps=1e-7, mode='exp', gamma=1, constant=12.8):

    center1 = (pred[:, :2] + pred[:, 2:]) / 2
    center2 = (target[:, :2] + target[:, 2:]) / 2

    whs = center1[:, :2] - center2[:, :2]

    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps  #

    w1 = pred[:, 2] - pred[:, 0] + eps
    h1 = pred[:, 3] - pred[:, 1] + eps
    w2 = target[:, 2] - target[:, 0] + eps
    h2 = target[:, 3] - target[:, 1] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance

    if mode == 'exp':
        normalized_wasserstein = torch.exp(-torch.sqrt(wasserstein_2) / constant)
        wloss = 1 - normalized_wasserstein

    elif mode == 'sqrt':
        wloss = torch.sqrt(wasserstein_2)

    elif mode == 'log':
        wloss = torch.log(wasserstein_2 + 1)

    elif mode == 'norm_sqrt':
        wloss = 1 - 1 / (gamma + torch.sqrt(wasserstein_2))
    else:
        #mode == 'w2':
        wloss = wasserstein_2

    return wloss.unsqueeze(1)


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).sum()
        return loss


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        nwd_loss = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        nwd_loss = (nwd_loss * weight).sum() / target_scores_sum

        loss_iou =  0.5 * loss_iou + nwd_loss

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)
