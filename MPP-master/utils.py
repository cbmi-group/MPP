import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets)
        
        intersection = (inputs * targets).sum(-1)                            
        dice = (2.*intersection + smooth)/(inputs.sum(-1) + targets.sum(-1) + smooth)
        
        return 1 - dice.mean()

def calculate_IoU(batch_out, label_stack):
    all_inter, all_union, all_pred, all_mask = 0, 0, 0, 0
    for k in range(batch_out.shape[0]):
        print(k)
        inter, iou = 0, 0
        pred, mask = batch_out[k].flatten(), label_stack[k].flatten()
        union = pred + mask
        union[union >1] = 1
        inter = (pred * mask)
        iou = (sum(inter) / sum(union))
        precision = (sum(inter)/ sum(pred))
        recall = (sum(inter) / sum(mask))
        f1 = 2 * precision * recall / (precision + recall)
        print(iou, precision, recall, f1)
    
        all_inter += sum(inter)
        all_union += sum(union)
        all_pred += sum(pred)
        all_mask += sum(mask)

    print('all')
    iou = all_inter / all_union
    precision = all_inter / all_pred
    recall = all_inter / all_mask
    f1 = 2 * precision * recall / (precision + recall)
    
    print(iou, precision, recall, f1)