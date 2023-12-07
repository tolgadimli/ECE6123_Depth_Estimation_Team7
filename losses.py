#Cite from: https://github.com/simonmeister/pytorch-mono-depth

import numpy as np
import torch
import torch.nn as nn
from math import log
# from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence



def _mask_input(input, mask=None):
    if mask is not None:
        input = input * mask
        count = torch.sum(mask).data[0]
    else:
        count = np.prod(input.size(), dtype=np.float32).item()
    return input, count


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class berHuLoss(nn.Module):
    def __init__(self):
        super(berHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c

        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()

        huber_mask = (diff > huber_c).detach()

        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2

        self.loss = torch.cat((diff, diff2)).mean()

        return self.loss

# class BerHuLoss(nn.Module):
#     def forward(self, input, target, mask=None):
#         x = input - target
#         abs_x = torch.abs(x)
#         c = torch.max(abs_x).item() / 5
#         leq = (abs_x <= c).float()
#         l2_losses = (x ** 2 + c ** 2) / (2 * c)
#         losses = leq * abs_x + (1 - leq) * l2_losses
#         losses, count = _mask_input(losses, mask)
#         return torch.sum(losses) / count


class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(size_average=False)

    def forward(self, input, target, mask=None):
        if mask is not None:
            loss = self.loss(input * mask, target * mask)
            count = torch.sum(mask).data[0]
            return loss / count

        count = np.prod(input.size(), dtype=np.float32).item()
        return self.loss(input, target) / count




class DistributionLogLoss(nn.Module):
    def __init__(self, distribution):
        super().__init__()
        self.distribution = distribution

    def forward(self, input, target, mask=None):
        d = self.distribution(*input)
        loss = d.log_loss(target)
        loss, count = _mask_input(loss, mask)
        return torch.sum(loss) / count


class RMSLoss(nn.Module):
    def forward(self, input, target, mask=None):
        loss = torch.pow(input - target, 2)
        loss, count = _mask_input(loss, mask)
        return torch.sqrt(torch.sum(loss) / count)


class RelLoss(nn.Module):
    def forward(self, input, target, mask=None):
        loss = torch.abs(input - target) / target
        loss, count = _mask_input(loss, mask)
        return torch.sum(loss) / count


class Log10Loss(nn.Module):
    def forward(self, input, target, mask=None):
        loss = torch.abs((torch.log(target) - torch.log(input)) / log(10))
        loss, count = _mask_input(loss, mask)
        return torch.sum(loss) / count


class TestingLosses(nn.Module):
    def __init__(self, scalar_losses):
        super().__init__()
        self.scalar_losses = nn.ModuleList(scalar_losses)

    def forward(self, input, target):
        scalars = [m(input, target) for m in self.scalar_losses]
        return torch.cat(scalars)


class OrdinalLoss(nn.Module):
    """
    Ordinal loss as defined in the paper "DORN for Monocular Depth Estimation".
    """

    def __init__(self, device):
        super(OrdinalLoss, self).__init__()
        self.device = device

    def forward(self, pred_softmax, target_labels):
        """
        :param pred_softmax:    predicted softmax probabilities P
        :param target_labels:   ground truth ordinal labels
        :return:                ordinal loss
        """
        N, C, H, W = pred_softmax.size() # C - number of discrete sub-intervals (= number of channels)

        K = torch.zeros((N, C, H, W), dtype=torch.int).to(self.device)
        for i in range(C):
            K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int).to(self.device)

        mask = (K <= target_labels).detach()
        
        loss = pred_softmax[mask].clamp(1e-8, 1e8).log().sum() + (1 - pred_softmax[~mask]).clamp(1e-8, 1e8).log().sum()
        loss /= -N * H * W
        return loss

# def relative_mse_loss(prediction: Tensor, target: Tensor, mask_zero: bool = False) -> float:
#     """
#     Compute MSE loss.

#     Args:
#         prediction (Tensor): Prediction.
#         target (Tensor): Target.
#         mask_zero (bool): Exclude zero values from the computation.

#     Returns:
#         float: MSE loss.
#     """

#     # let's only compute the loss on non-null pixels from the ground-truth depth-map
#     if mask_zero:
#         non_zero_mask = target > 0
#         masked_input = prediction[non_zero_mask]
#         masked_target = target[non_zero_mask]
#     else:
#         masked_input = prediction
#         masked_target = target

#     # Prediction MSE loss
#     pred_mse = functional.mse_loss(masked_input, masked_target)

#     # Self MSE loss for mean target
#     target_mse = functional.mse_loss(masked_target, torch.ones_like(masked_target) * torch.mean(masked_target))

#     return pred_mse / target_mse * 100




class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=False): #we set this to false
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


# class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
#     def __init__(self):
#         super().__init__()
#         self.name = "ChamferLoss"

#     def forward(self, bins, target_depth_maps):
#         bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
#         n, p = bin_centers.shape
#         input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
#         # n, c, h, w = target_depth_maps.shape

#         target_points = target_depth_maps.flatten(1)  # n, hwc
#         mask = target_points.ge(1e-3)  # only valid ground truth points
#         target_points = [p[m] for p, m in zip(target_points, mask)]
#         target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
#         target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

#         loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
#         return loss