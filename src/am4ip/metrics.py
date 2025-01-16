import torch
from abc import ABC, abstractmethod


class IQMetric(ABC):
    """Abstract IQ metric class.
    """
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class FullReferenceIQMetric(IQMetric):
    """
    Abstract class to implement full-reference IQ metrics.
    """

    @abstractmethod
    def __call__(self, im: torch.Tensor, im_ref: torch.Tensor, *args) -> torch.Tensor:
        """
        Compute the metric over im and

        :param im: Batch of distorted images. Size = N x C x H x W
        :param im_ref: Batch of reference images. Size = N x C x H x W
        :return: IQ metric for each pair. Size = N
        """
        raise NotImplementedError


class NoReferenceIQMetric(IQMetric):
    """
    Abstract class to implement no-reference IQ metrics.
    """

    @abstractmethod
    def __call__(self, im: torch.Tensor, *args) -> torch.Tensor:
        """
        Compute the metric over im and

        :param im: Batch of distorted images. Size = N x C x H x W
        :return: IQ metric for each pair. Size = N
        """
        raise NotImplementedError


class NormalizedMeanAbsoluteError(FullReferenceIQMetric):
    """
    Compute normalized mean absolute error (MAE) on images.

    Note that nMAE is a distortion metric, not a quality metric. This means that it 
    should be negatively correlated with Mean Opinion Scores.
    """
    def __init__(self, norm=255.):
        super(NormalizedMeanAbsoluteError, self).__init__(name="nMAE")
        self.norm = norm

    def __call__(self, im: torch.Tensor, im_ref: torch.Tensor, *args) -> torch.Tensor:
        return torch.mean(torch.abs(im - im_ref) / self.norm, dim=[1, 2, 3])  # Average over C x H x W

class MeanIntersectionOverUnion(FullReferenceIQMetric):
    def __init__(self, norm=255.):
        super(MeanIntersectionOverUnion, self).__init__(name="MeanIntersectionOverUnion")
        self.norm = norm

    def __call__(self, im: torch.Tensor, im_ref: torch.Tensor, *args) -> torch.Tensor:
        unique_classes = torch.unique(torch.cat([im, im_ref]))
        nClasses = unique_classes.numel()

        intersection = torch.zeros(nClasses, device=im.device, dtype=torch.float32)
        union = torch.zeros(nClasses, device=im.device, dtype=torch.float32)

        for idx, c in enumerate(unique_classes):
            pred_c = (im == c)
            ref_c = (im_ref == c)
            intersection[idx] += torch.sum(pred_c & ref_c).float()
            union[idx] += torch.sum(pred_c | ref_c).float()

        IoU = intersection / (union + 1e-6)
        mIoU = IoU.mean()
        return mIoU
    
# Aliases
nMAE = NormalizedMeanAbsoluteError
mIoU = MeanIntersectionOverUnion()