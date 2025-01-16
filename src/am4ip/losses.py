
import torch
# noinspection PyProtectedMember
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=None, smooth=1e-6):
        """
        Initialize the Dice Loss.

        :param ignore_index: Index of the class to ignore (e.g., for ignoring the background class).
        :param smooth: Smoothing factor to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Compute the Dice Loss.

        :param preds: Model predictions (tensor of shape [B, C, H, W], with logits or probabilities).
        :param targets: Ground truth (tensor of shape [B, H, W], with class indices).
        :return: Scalar value of the Dice Loss.
        """
        # Convert predictions to probabilities if they are logits
        if preds.shape[1] > 1:  # Check if the tensor has multiple classes
            preds = F.softmax(preds, dim=1)  # [B, C, H, W]

        num_classes = preds.shape[1]
        dice_loss = 0.0
        count = 0

        # Loop over each class
        for class_idx in range(num_classes):
            if self.ignore_index is not None and class_idx == self.ignore_index:
                continue  # Skip the specified class

            # Create masks for the current class
            pred_class = preds[:, class_idx, :, :]  # [B, H, W]
            target_class = (targets == class_idx).float()  # [B, H, W]

            # Compute intersection and union for the class
            intersection = (pred_class * target_class).sum(dim=(1, 2))  # [B]
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))  # [B]

            # Compute Dice coefficient (with smoothing)
            dice = (2 * intersection + self.smooth) / (union + self.smooth)  # [B]
            dice_loss += 1 - dice.mean()  # Average over the batch
            count += 1

        # Average Dice Loss over all classes
        dice_loss = dice_loss / count if count > 0 else 1.0
        return dice_loss
    
class DiceLoss2(nn.Module):
    def __init__(self):
        """
        TotalLoss: Combines Total Variation Loss and Asymmetric Loss.

        Args:
            lambda_tv (float): Weight for the Total Variation loss.
            lambda_asymm (float): Weight for the Asymmetric loss.
            alpha (float): Factor for asymmetry in the Asymmetric loss.
        """
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true, bg=0):
        loss = 0.0
        cpt1 = 0

        for i in range(y_true.shape[0]):
            tmp_loss = 0.0
            epsilon = 0.00000000001
            cpt2 = 0

            for l in range(bg,y_true.shape[1]):
                if torch.sum(y_true[i,l]).item() == 0:
                    continue

                a = y_true[i,l,:,:]
                b = y_pred[i,l,:,:]
                tmp_loss += (2*torch.sum(a[:]*b[:]) / (torch.sum(a[:]) + torch.sum(b[:]) + epsilon) )
                cpt2 += 1

            if (cpt2-bg) != 0:
                tmp_loss = tmp_loss/(cpt2-bg)
            else:
                continue

            loss += tmp_loss
            cpt1 += 1

        if cpt1 == 0:
            return -1

        return 1 - loss/cpt1