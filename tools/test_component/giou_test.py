import torch

def generalized_iou_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0

    area1 = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
    area2 = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
    print("area1, area2:", area1, area2)

    overlap_lt = torch.max(pred[:, :2], target[:, :2])  # [n, 2]
    overlap_rb = torch.min(pred[:, 2:], target[:, 2:])
    overlap_wh = (overlap_rb - overlap_lt + 1).clamp(min=0)
    overlap = overlap_wh[:, 0] * overlap_wh[:, 1]
    print("overlap:", overlap)

    convex_lt = torch.min(pred[:, :2], target[:, :2])
    convex_rb = torch.max(pred[:, 2:], target[:, 2:])
    convex_wh = (convex_rb - convex_lt + 1).clamp(min=0)
    convex = convex_wh[:, 0] * convex_wh[:, 1]
    print("convex:", convex)

    unions = area1 + area2 - overlap
    ious = torch.where(overlap == 0.0,
                       torch.zeros_like(overlap, device=overlap.device), overlap / unions)
    gious = ious - (convex - unions) / convex.clamp(min=1e-10)  # [n]

    return 1 - gious  # (1 - gious).sum()

b1 = torch.Tensor([[2, 1, 5, 3]])
b2 = torch.Tensor([[4, 2, 7, 5]])
print(generalized_iou_loss(b1, b2))