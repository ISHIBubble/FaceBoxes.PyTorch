import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import match
from data import cfg
import math

GPU = cfg['gpu_train']


class FocalCIoULoss(nn.Module):
    """
    FaceBoxes Loss with Focal Loss (classification) and CIoU Loss (localization).
    
    Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    CIoU Loss: L_CIoU = 1 - IoU + ρ²(b, b^gt)/c² + αv
    
    where:
    - ρ² is the squared Euclidean distance between centers
    - c² is the squared diagonal of the smallest enclosing box
    - v measures aspect ratio consistency
    - α is a trade-off parameter
    
    Args:
        num_classes: Number of classes (2 for face detection)
        overlap_thresh: IoU threshold for matching (default: 0.35)
        prior_for_matching: Use priors for matching
        bkg_label: Background label index (default: 0)
        neg_mining: Whether to use hard negative mining (not used with focal loss)
        neg_pos: Negative to positive ratio (not used with focal loss)
        neg_overlap: Overlap threshold for negatives
        encode_target: Whether to encode targets
        focal_alpha: Focal loss alpha parameter (default: 0.25)
        focal_gamma: Focal loss gamma parameter (default: 2.0)
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, 
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 focal_alpha=0.25, focal_gamma=2.0):
        super(FocalCIoULoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.variance = [0.1, 0.2]
        
        # Focal Loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def forward(self, predictions, priors, targets):
        """
        Compute Focal Loss + CIoU Loss.
        
        Args:
            predictions (tuple): (loc_data, conf_data) from FaceBoxes
                - loc_data: [batch, num_priors, 4] - encoded box predictions
                - conf_data: [batch, num_priors, num_classes] - class logits
            priors: [num_priors, 4] - prior boxes in center-size format
            targets: List of [num_objs, 5] tensors (x1, y1, x2, y2, label)
            
        Returns:
            loss_l: Localization loss (CIoU)
            loss_c: Classification loss (Focal)
        """
        loc_data, conf_data = predictions
        num = loc_data.size(0)
        num_priors = priors.size(0)
        
        # Match priors to ground truth
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, 
                  labels, loc_t, conf_t, idx)
        
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        
        # =====================
        # CIoU Loss (Localization)
        # =====================
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        
        # Get positive predictions and targets
        loc_p = loc_data[pos_idx].view(-1, 4)  # Encoded predictions
        loc_t_pos = loc_t[pos_idx].view(-1, 4)  # Encoded targets
        
        if loc_p.size(0) > 0:
            # Get the corresponding priors for positive samples
            pos_priors = priors.unsqueeze(0).expand(num, num_priors, 4)
            pos_priors = pos_priors[pos_idx[:, :, :1].expand_as(pos_priors)].view(-1, 4)
            
            # Decode predictions and targets to corner format for CIoU
            pred_boxes = self._decode_boxes(loc_p, pos_priors)
            target_boxes = self._decode_boxes(loc_t_pos, pos_priors)
            
            # Compute CIoU loss
            loss_l = self._ciou_loss(pred_boxes, target_boxes)
        else:
            loss_l = torch.tensor(0.0, device=loc_data.device)
        
        # =====================
        # Focal Loss (Classification)
        # =====================
        # Reshape for focal loss computation
        batch_conf = conf_data.view(-1, self.num_classes)
        conf_t_flat = conf_t.view(-1)
        
        loss_c = self._focal_loss(batch_conf, conf_t_flat)
        
        # Normalize by number of positives
        N = max(num_pos.data.sum().float(), 1.0)
        loss_l = loss_l / N
        loss_c = loss_c / N
        
        return loss_l, loss_c
    
    def _decode_boxes(self, loc, priors):
        """
        Decode encoded boxes to corner format (x1, y1, x2, y2).
        
        Args:
            loc: Encoded box predictions [N, 4]
            priors: Prior boxes in center-size format [N, 4]
            
        Returns:
            boxes: Decoded boxes in corner format [N, 4]
        """
        # Decode center and size
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * self.variance[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * self.variance[1])
        ), 1)
        
        # Convert from center-size to corner format
        boxes_corner = torch.cat((
            boxes[:, :2] - boxes[:, 2:] / 2,  # x1, y1
            boxes[:, :2] + boxes[:, 2:] / 2   # x2, y2
        ), 1)
        
        return boxes_corner
    
    def _ciou_loss(self, pred_boxes, target_boxes):
        """
        Compute Complete IoU (CIoU) Loss.
        
        L_CIoU = 1 - IoU + ρ²(b, b^gt)/c² + αv
        
        where:
        - ρ² is the squared Euclidean distance between centers
        - c² is the squared diagonal of the smallest enclosing box
        - v measures aspect ratio consistency
        - α is a trade-off parameter
        
        Args:
            pred_boxes: Predicted boxes [N, 4] in (x1, y1, x2, y2) format
            target_boxes: Target boxes [N, 4] in (x1, y1, x2, y2) format
            
        Returns:
            CIoU loss (scalar)
        """
        eps = 1e-7
        
        # Extract coordinates
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(dim=1)
        gt_x1, gt_y1, gt_x2, gt_y2 = target_boxes.unbind(dim=1)
        
        # Predicted box dimensions
        pred_w = (pred_x2 - pred_x1).clamp(min=eps)
        pred_h = (pred_y2 - pred_y1).clamp(min=eps)
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        
        # Ground truth box dimensions
        gt_w = (gt_x2 - gt_x1).clamp(min=eps)
        gt_h = (gt_y2 - gt_y1).clamp(min=eps)
        gt_cx = (gt_x1 + gt_x2) / 2
        gt_cy = (gt_y1 + gt_y2) / 2
        
        # Intersection area
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Union area
        pred_area = pred_w * pred_h
        gt_area = gt_w * gt_h
        union_area = pred_area + gt_area - inter_area + eps
        
        # IoU
        iou = inter_area / union_area
        
        # Enclosing box (smallest box containing both boxes)
        enclose_x1 = torch.min(pred_x1, gt_x1)
        enclose_y1 = torch.min(pred_y1, gt_y1)
        enclose_x2 = torch.max(pred_x2, gt_x2)
        enclose_y2 = torch.max(pred_y2, gt_y2)
        
        # Diagonal of enclosing box squared
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        c2 = enclose_w ** 2 + enclose_h ** 2 + eps  # c² (diagonal squared)
        
        # Center distance squared (ρ²)
        rho2 = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
        
        # Aspect ratio consistency term (v)
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(gt_w / gt_h) - torch.atan(pred_w / pred_h), 2
        )
        
        # Trade-off parameter α
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        
        # CIoU
        ciou = iou - (rho2 / c2) - alpha * v
        
        # CIoU Loss
        loss = 1 - ciou
        
        return loss.sum()
    
    def _focal_loss(self, pred, target):
        """
        Compute Focal Loss for classification.
        
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        
        Args:
            pred: Class predictions [N, num_classes] (logits)
            target: Ground truth class labels [N]
            
        Returns:
            Focal loss (scalar)
        """
        # Convert logits to probabilities
        pred_softmax = F.softmax(pred, dim=1)
        
        # Get the probability for the true class
        # Create one-hot encoding
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Get p_t (probability of true class)
        p_t = (pred_softmax * target_one_hot).sum(dim=1)
        
        # Compute focal weight: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # Compute α_t: α for positive class, (1-α) for background
        # target > 0 means positive (face), target == 0 means background
        alpha_t = torch.where(
            target > 0,
            torch.tensor(self.focal_alpha, device=pred.device, dtype=pred.dtype),
            torch.tensor(1 - self.focal_alpha, device=pred.device, dtype=pred.dtype)
        )
        
        # Compute cross-entropy: -log(p_t)
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return focal_loss.sum()


# Alias for backward compatibility with train.py interface
MultiBoxLoss = FocalCIoULoss
