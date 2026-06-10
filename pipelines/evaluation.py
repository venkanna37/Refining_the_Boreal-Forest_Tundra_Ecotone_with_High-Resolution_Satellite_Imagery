import os
import torch
import rasterio
import numpy as np
from glob import glob
import torch.nn as nn
from pipelines import network
from torchmetrics.classification import BinaryStatScores
from pipelines import datagen, network, network_elu, network_elu_bn


def estimate_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
    f1 = 2 * precision * recall / (precision + recall + 0.000001)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, f1, accuracy


# function that take dictionary of models and gives the final mask
@torch.no_grad()
def ensemble_union_predict(models,
                           input_patch,
                           ensemble_mode="majority"):
    """
    Predict from multiple models
    """

    individual_masks = []

    for model in models.values() if isinstance(models, dict) else models:
        logits = model(input_patch)

        # Handle (B, H, W) → (B, 1, H, W)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        probs = torch.sigmoid(logits)

        # binary_mask = (probs > 0.5).to(torch.uint8)
        individual_masks.append(probs)

    stacked = torch.stack(individual_masks, dim=0)
    if ensemble_mode == "majority":
        # MAJORITY
        mask = torch.mean(stacked, dim=0)
        mask = (mask > 0.5).to(torch.uint8)
    elif ensemble_mode == "intersection":
        # INTERSECTION
        mask = stacked > 0.5
        mask = torch.all(mask, dim=0).to(torch.uint8)
    elif ensemble_mode == "atleast1":
        # ATLEAST ONE
        mask = stacked > 0.5
        mask = (torch.sum(mask, dim=0) >= 1).to(torch.uint8)
    elif ensemble_mode == "atleast2":
        # ATLEAST TWO
        mask = stacked > 0.5
        mask = (torch.sum(mask, dim=0) >= 2).to(torch.uint8)
    elif ensemble_mode == "atleast3":
        # ATLEAST THREEE
        mask = stacked > 0.5
        mask = (torch.sum(mask, dim=0) >= 3).to(torch.uint8)
    elif ensemble_mode == "atleast4":
        # ATLEAST FOUR
        mask = stacked > 0.5
        mask = (torch.sum(mask, dim=0) >= 4).to(torch.uint8)
    else:
        raise NotImplementedError(f"ensemble_mode={ensemble_mode} not implemented")

    return mask


# function that takes the image and give the prediction
def apply_model_on_geotiff(
        geotiff_path,
        checkpoint_path,
        device='cpu',
        rescale_value = 2000,
        ensemble_mode='majority',
        model_name='unet_elu'):

    models = {}
    for i, path in enumerate(checkpoint_path):
        # Setup model
        if model_name == 'unet_elu':
            model = network_elu.TinyUNet()
        elif model_name == 'unet_elu_bn':
            model = network_elu_bn.TinyUNet()
        elif model_name == 'unet_relu_bn':
            model = network.TinyUNet()
        else:
            raise Exception("Unknown model name")
        # model = network.TinyUNet()
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
        models[f"model_{i + 1}"] = model

    with rasterio.open(geotiff_path) as src:
        image = src.read(1).astype(np.float32) / rescale_value
        image = np.expand_dims(image, 0)

    image = torch.from_numpy(image)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    _, c, h, w = image.shape
    # image = nn.ReflectionPad2d(47)(image)
    with torch.no_grad():
        patch = image.to(device)
        if len(checkpoint_path) > 1:
            pred = ensemble_union_predict(models, patch, ensemble_mode)
        else:
            model = models['model_1']
            logits = model(patch)
            pred = (logits > 0).int()

    return pred[0, 0]


class ArcticEvaluation:
    def __init__(self, **kwargs):
        self.images = sorted(glob(os.path.join(kwargs['image_dir'], '*.tif')))
        self.labels = sorted(glob(os.path.join(kwargs['label_dir'], '*.tif')))
        self.pretrained_model = kwargs['pretrained_model']
        self.ensemble = kwargs['ensemble']
        self.set_name = kwargs['set_name']
        self.model_name = kwargs['model_name']
        self.ensemble_mode = kwargs['ensemble_mode']
        self.slice = 7 if self.set_name == 'test' else 24

        # print the parameters as a dictionary
        self.kwargs = kwargs
        print("\n -----Training parameters-----")
        for key, value in self.kwargs.items():
            print(f"  {key:20s}: {value}")
        print("----------------------------- \n")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):

        metrics = BinaryStatScores().to(self.device)
        individual_metrics = BinaryStatScores().to(self.device)

        for image_path, label_path in zip(self.images, self.labels):
            assert os.path.basename(image_path)[:self.slice] == os.path.basename(label_path)[:self.slice],'image != label'
            # get prediction mask
            y_pred = apply_model_on_geotiff(image_path,
                                            self.pretrained_model,
                                            device=self.device,
                                            ensemble_mode=self.ensemble_mode,
                                            model_name=self.model_name)

            # get gound truth mask
            with rasterio.open(label_path) as src:
                y_true = src.read(1)
                y_true[y_true > 1] = 0
                y_true = y_true[47:-47, 47:-47]
                y_true = torch.from_numpy(y_true).to(self.device)

            # y_pred = y_pred[1:-1, 1:-1] #fixme
            metrics.update(y_pred, y_true)
            if self.set_name == 'test':
                individual_metrics.reset()
                individual_metrics.update(y_pred, y_true)
                tp, fp, tn, fn, sup = individual_metrics.compute()
                pre, rec, f1, acc = estimate_metrics(tp, fp, tn, fn)
                tp, fp, tn, fn = round(tp.item(),2), round(fp.item(),2), round(tn.item(),2), round(fn.item(),2)
                pre, rec, f1, acc = round(pre.item(),2), round(rec.item(),2), round(f1.item(),2), round(acc.item(),2)

                print(f'{os.path.basename(image_path)[:14]:15s} {str(tp):10s} {str(tn):10s} {str(fp):10s} {str(fn):10s}'
                      f' {str(pre):7s} {str(rec):7s} {str(f1):7s} {acc}')

        tp, fp, tn, fn, sup = metrics.compute()
        pre, rec, f1, acc = estimate_metrics(tp, fp, tn, fn)
        tp, fp, tn, fn = round(tp.item(), 2), round(fp.item(), 2), round(tn.item(), 2), round(fn.item(), 2)
        pre, rec, f1, acc = round(pre.item(), 2), round(rec.item(), 2), round(f1.item(), 2), round(acc.item(), 2)

        print(f'{"ALL":15s} {str(tp):10s} {str(tn):10s} {str(fp):10s} {str(fn):10s}'
              f' {str(pre):7s} {str(rec):7s} {str(f1):7s} {acc}')



