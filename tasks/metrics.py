import os
import torch
import rasterio
from glob import glob
import numpy as np
from torchmetrics.classification import BinaryStatScores
import matplotlib.pyplot as plt

def estimate_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
    f1 = 2 * precision * recall / (precision + recall + 0.000001)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, f1, accuracy


image_213_433 = '/home/venky/Documents/projects/arctic/data/test/images/213_433_spring.tif'
labels =  sorted(glob('/home/venky/Documents/projects/arctic/data/test/labels/*.tif'))
predictions = sorted(glob('/home/venky/Documents/projects/data/arctic/test_pred_from_mosaic/*.tif'))

metrics = BinaryStatScores().to('cpu')
individual_metrics = BinaryStatScores().to('cpu')

for label, prediction in zip(labels, predictions):
    # print(os.path.basename(label), os.path.basename(prediction))
    assert os.path.basename(label)[:7] == os.path.basename(prediction)[:7], 'site ids are not same'
    with rasterio.open(prediction) as src:
        y_pred = src.read(1)

    with rasterio.open(label) as src:
        y_true = src.read(1)

    if '213_433' in os.path.basename(label):
        with rasterio.open(image_213_433) as src:
            mask = src.read(1)
            valid_mask = mask != 0
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

    y_true = torch.from_numpy(y_true).to('cpu')
    y_pred = torch.from_numpy(y_pred).to('cpu')
    metrics.update(y_pred, y_true)

    individual_metrics.reset()
    individual_metrics.update(y_pred, y_true)

    tp, fp, tn, fn, sup = individual_metrics.compute()
    pre, rec, f1, acc = estimate_metrics(tp, fp, tn, fn)
    tp, fp, tn, fn = round(tp.item(), 2), round(fp.item(), 2), round(tn.item(), 2), round(fn.item(), 2)
    pre, rec, f1, acc = round(pre.item(), 2), round(rec.item(), 2), round(f1.item(), 2), round(acc.item(), 2)

    print(f'{os.path.basename(prediction)[:14]:15s} {str(tp):10s} {str(tn):10s} {str(fp):10s} {str(fn):10s}'
          f' {str(pre):7s} {str(rec):7s} {str(f1):7s} {acc}')

tp, fp, tn, fn, sup = metrics.compute()
pre, rec, f1, acc = estimate_metrics(tp, fp, tn, fn)
tp, fp, tn, fn = round(tp.item(), 2), round(fp.item(), 2), round(tn.item(), 2), round(fn.item(), 2)
pre, rec, f1, acc = round(pre.item(), 2), round(rec.item(), 2), round(f1.item(), 2), round(acc.item(), 2)

print(f'{"ALL":15s} {str(tp):10s} {str(tn):10s} {str(fp):10s} {str(fn):10s}'
  f' {str(pre):7s} {str(rec):7s} {str(f1):7s} {acc}')