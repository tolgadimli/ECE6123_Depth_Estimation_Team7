# Copyright Enlens LLC, 2021. All rights reserved.
#
# This software is licensed under the terms of the "BANet Monocular Depth Prediction" licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Author: Denis Girard (denis.girard@enlens.net)

import torch
import torch.nn.functional as functional
from torch import Tensor
import numpy as np


def calc_metrics(prediction, target):

    assert prediction.shape == target.shape
    pred255, target255 = prediction, target

    abs_rel = relative_mae_loss( pred255, target255 )
    sqr_rel = relative_mse_loss( pred255, target255 )

    rmse_linear = rmse_linear_loss(pred255, target255)
    rmse_log = rmse_log_loss(pred255, target255)
    si_log = silog_loss(pred255, target255)

    a1 = threshold_accuracy( pred255, target255, threshold = 1.25, level = 1 )
    a2 = threshold_accuracy( pred255, target255, threshold = 1.25, level = 2 )
    a3 = threshold_accuracy( pred255, target255, threshold = 1.25, level = 3 )

    return [abs_rel.item(), sqr_rel.item(), rmse_linear.item(), rmse_log.item(), si_log.item(), a1.item(), a2.item(), a3.item()]




def threshold_accuracy( prediction, target, threshold = 1.25, level = 1 ):


    a, b= prediction / target, target/prediction
    t = torch.maximum(a, b)
    corrects = torch.sum( t < threshold ** level )
    return 100 * corrects / np.prod(t.shape)


def squared_rd(prediction, target):
    loss = (prediction-target) * (prediction-target) / target
    return loss.mean()

def abs_rd(prediction, target):
    loss = torch.abs(prediction - target) / target
    return loss.mean()

def relative_mse_loss(prediction: Tensor, target: Tensor, mask_zero: bool = False) -> float:
    """
    Compute MSE loss.

    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        mask_zero (bool): Exclude zero values from the computation.

    Returns:
        float: MSE loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    if mask_zero:
        non_zero_mask = target > 0
        masked_input = prediction[non_zero_mask]
        masked_target = target[non_zero_mask]
    else:
        masked_input = prediction
        masked_target = target

    # Prediction MSE loss
    pred_mse = functional.mse_loss(masked_input, masked_target)

    # Self MSE loss for mean target
    # target_mse = functional.mse_loss(masked_target, torch.ones_like(masked_target) * torch.mean(masked_target))
    target_mse = functional.mse_loss(masked_target, torch.zeros_like(masked_target) )

    return pred_mse / target_mse


def relative_mae_loss(prediction: Tensor, target: Tensor, mask_zero: bool = True):
    """
    Compute MAE loss.

    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        mask_zero (bool): Exclude zero values from the computation.

    Returns:
        float: MAE loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    if mask_zero:
        non_zero_mask = target > 0
        masked_input = prediction[non_zero_mask]
        masked_target = target[non_zero_mask]
    else:
        masked_input = prediction
        masked_target = target

    # Prediction MSE loss
    pred_mae = functional.l1_loss(masked_input, masked_target)

    # Self MSE loss for mean target
    # target_mae = functional.l1_loss(masked_target, torch.ones_like(masked_target) * torch.mean(masked_target))
    target_mae = functional.l1_loss(masked_target, torch.zeros_like(masked_target))

    return pred_mae / target_mae


def rmse_linear_loss(prediction: Tensor, target: Tensor) -> float:

    norm_sqr = torch.norm( prediction - target ) ** 2
    return torch.sqrt( norm_sqr / np.prod(prediction.shape) )


def rmse_log_loss(prediction: Tensor, target: Tensor) -> float:

    norm_sqr = torch.norm( torch.log(prediction + 1e-7) - torch.log(target + 1e-7) ) ** 2
    return torch.sqrt( norm_sqr / np.prod(prediction.shape)  )


def silog_loss(prediction: Tensor, target: Tensor, variance_focus: float = 1) -> float:
    """
    Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    more information about scale-invariant loss.

    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        variance_focus (float): Variance focus for the SILog computation.

    Returns:
        float: SILog loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    non_zero_mask = (target > 0) & (prediction > 0)

    # SILog
    d = torch.log(prediction[non_zero_mask]) - torch.log(target[non_zero_mask])
    return torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2))

# def silog_loss(prediction: Tensor, target: Tensor) -> float:
#     """
#     Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
#     more information about scale-invariant loss.

#     Args:
#         prediction (Tensor): Prediction.
#         target (Tensor): Target.
#         variance_focus (float): Variance focus for the SILog computation.

#     Returns:
#         float: SILog loss.
#     """

#     # let's only compute the loss on non-null pixels from the ground-truth depth-map
#     # non_zero_mask = (target > 0) & (prediction > 0)

#     # SILog
#     n = prediction.shape[0]
#     d = torch.log(prediction + 1e-7) - torch.log(target + 1e-7)
#     return torch.sum( d**2) / n - (torch.sum(d) ** 2) / (n**2)

# pred, target = torch.rand(16,1,100,100), torch.rand(16,1,100,100)
# print(calc_metrics(pred, target))
# print(calc_metrics(255*pred, 255*target))

