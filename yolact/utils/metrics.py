import torch
from torch import Tensor


def Jaccard(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Jaccard coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        union = torch.sum(input) + torch.sum(target) - inter

        return (inter + epsilon) / (union + epsilon)
    else:
        # compute and average metric for each batch element
        jaccard = 0
        for i in range(input.shape[0]):
            jaccard += Jaccard(input[i, ...], target[i, ...])
        return jaccard / input.shape[0]


def multiclass_Jaccard(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    jaccard = 0
    for channel in range(input.shape[1]):
        jaccard += Jaccard(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return jaccard / input.shape[1]