import torch.nn as nn
import torch.nn.init as init

def init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MetricsCalculator:
    """Calculate metrics from accumulated TP, FP, FN, TN"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred, target):
        pred, target = pred.flatten(), target.flatten()
        self.tp += ((pred == 1) & (target == 1)).sum().item()
        self.fp += ((pred == 1) & (target == 0)).sum().item()
        self.fn += ((pred == 0) & (target == 1)).sum().item()
        self.tn += ((pred == 0) & (target == 0)).sum().item()

    def compute(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + 1e-8)
        iou = self.tp / (self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) > 0 else 0.0

        return {
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': accuracy, 'iou': iou,
            'tp': self.tp, 'fp': self.fp, 'fn': self.fn, 'tn': self.tn
        }