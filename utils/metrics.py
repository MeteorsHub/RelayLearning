from typing import Tuple

import numpy as np
import scipy.stats as st
import torch
from torch import Tensor
from torchmetrics import IoU, Metric
from torchmetrics.utilities.distributed import reduce

from utils.surface_distance.metrics import compute_robust_hausdorff, compute_surface_distances


def confidence_interval_boostrap(samples: torch.Tensor,
                                 stats_func: callable = torch.mean, ci=0.95, num_boostrap=1000):
    assert isinstance(samples, (torch.Tensor, np.ndarray))

    n = len(samples)

    device = samples.device
    samples_np = samples.detach().cpu().numpy()

    rng = np.random.default_rng(seed=1234)
    mean_stats = stats_func(samples)

    stats = []
    for _ in range(num_boostrap):
        samples_ = rng.choice(samples_np, size=n, replace=True)
        samples_ = torch.tensor(samples_, device=device)
        stats_ = stats_func(samples_)
        stats.append(stats_.detach().cpu().numpy())
    low = 100 * (1 - ci) / 2
    high = low + 100 * ci
    interval = (np.percentile(stats, low), np.percentile(stats, high))
    interval = (torch.tensor(interval[0], device=device), torch.tensor(interval[1], device=device))

    return mean_stats, interval


def confidence_interval_t(samples: torch.Tensor, ci=0.95):
    assert samples.ndim == 1
    n = len(samples)
    dof = n - 1
    mean, se = samples.mean(), samples.std(unbiased=True) / np.sqrt(n)
    interval = st.t.interval(alpha=ci, df=dof, loc=mean.item(), scale=se.item())
    interval = torch.tensor(interval, device=samples.device)
    return mean, interval


class Dice(IoU):
    """ This is voxel-wise calculation."""
    def __init__(self, num_classes, class_id=None, bg=False, **kwargs):
        if class_id is not None:
            assert class_id < num_classes
        self.class_id = class_id
        self.bg = bg
        super().__init__(num_classes, reduction='none', **kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.num_classes == 2 and preds.ndim != target.ndim:
            assert preds.shape[1] == 2
            preds, target = preds[:, 1].reshape(-1), target.reshape(-1)
        super().update(preds, target)

    def compute(self) -> Tensor:
        iou = super().compute()
        dice = 2 * iou / (1 + iou)
        if self.class_id is not None:
            return dice[self.class_id]
        if self.bg:
            return torch.mean(dice)
        else:
            return torch.mean(dice[1:])


class DiceSampleWise(Metric):
    def __init__(self, class_id=None, bg=False, **kwargs):
        super().__init__(**kwargs)
        self.add_state('all_dice', default=[], dist_reduce_fx='cat')
        self.class_id = class_id
        self.bg = bg

    def update(self, preds: torch.Tensor, target: torch.Tensor, metas={}) -> None:
        assert preds.ndim == target.ndim + 1
        for i in range(len(preds)):
            dice = dice_score(preds[i:i + 1], target[i:i + 1], bg=self.bg, nan_score=1.0, reduction='none')
            if self.class_id is not None:
                assert self.class_id - 1 < len(dice)
                dice = dice[self.class_id - 1]
            else:
                dice = torch.mean(dice)
            self.all_dice.append(dice)

    def compute(self):
        return self.dice_all_samples

    @property
    def dice_all_samples(self):
        if torch.is_tensor(self.all_dice):
            return self.all_dice
        return torch.stack(self.all_dice)


def stat_scores(
        preds: Tensor,
        target: Tensor,
        class_index: int,
        argmax_dim: int = 1) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, dim=argmax_dim)

    tp = ((preds == class_index) * (target == class_index)).to(torch.long).sum()
    fp = ((preds == class_index) * (target != class_index)).to(torch.long).sum()
    tn = ((preds != class_index) * (target != class_index)).to(torch.long).sum()
    fn = ((preds != class_index) * (target == class_index)).to(torch.long).sum()
    sup = (target == class_index).to(torch.long).sum()

    return tp, fp, tn, fn, sup


def dice_score(
        preds: Tensor,
        target: Tensor,
        bg: bool = False,
        nan_score: float = 1.0,
        reduction: str = "elementwise_mean") -> Tensor:
    num_classes = preds.shape[1]
    bg_inv = 1 - int(bg)
    scores = torch.zeros(num_classes - bg_inv, device=preds.device, dtype=torch.float32)
    for i in range(bg_inv, num_classes):
        tp, fp, _, fn, _ = stat_scores(preds=preds, target=target, class_index=i)
        denom = (2 * tp + fp + fn).to(torch.float)
        # nan result
        score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else nan_score
        scores[i - bg_inv] += score_cls
    return reduce(scores, reduction=reduction)


class HausdorffDistance(Metric):
    def __init__(self, class_id=1, store_all=False, inf_score=100, **kwargs):
        super().__init__(**kwargs)
        if not store_all:
            self.add_state('total_hd', default=torch.tensor(0., dtype=torch.double), dist_reduce_fx='sum')
            self.add_state('total_num', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')
        else:
            self.add_state('all_hd', default=[], dist_reduce_fx='cat')
        self.class_id = class_id
        self.store_all = store_all
        self.inf_score = inf_score

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        device = pred.device
        if pred.ndim == target.ndim + 1:  # NCHW..., NHW...
            if self.class_id is not None:
                assert pred.shape[1] > self.class_id
            pred = pred.argmax(dim=1)
        elif pred.ndim != target.ndim:  # no NHW...
            raise RuntimeError('Pred and target to the HausdorffDistance is invalid')

        assert pred.dtype in [torch.int, torch.long]
        pred = pred == self.class_id
        target = target == self.class_id

        # pred: NHW...; target: NHW...
        assert pred.shape == target.shape, 'pred and target have different shape in HausdorffDistance'

        for p, t in zip(pred, target):
            p, t = p.cpu().numpy(), t.cpu().numpy()
            surface_distances = compute_surface_distances(t, p, spacing_mm=[1] * p.ndim)
            if np.sum(surface_distances["surfel_areas_gt"]) == 0 and np.sum(
                    surface_distances["surfel_areas_pred"]) == 0:
                hd95 = 0.
            elif np.sum(surface_distances["surfel_areas_gt"]) == 0 or np.sum(
                    surface_distances["surfel_areas_pred"]) == 0:
                hd95 = self.inf_score
            else:
                hd95 = compute_robust_hausdorff(surface_distances, percent=95.)
            if hd95 == np.Inf:
                hd95 = self.inf_score
            hd95 = torch.tensor(hd95, dtype=torch.double, device=device)
            if self.store_all:
                self.all_hd.append(hd95)
            else:
                self.total_hd += hd95
                self.total_num += 1

    def compute(self) -> torch.Tensor:
        if self.store_all:
            if torch.is_tensor(self.all_hd):
                return self.all_hd
            return torch.stack(self.all_hd)
        else:
            return self.total_hd / (self.total_num.double() + 1e-8)
