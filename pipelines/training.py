"""
Training UNet with WandB logging and comprehensive metrics
"""

import os
import torch
import wandb
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.data import DataLoader

from pipelines import datagen, network
from pipelines import utils


def tversky(y_t, y_pred, y_w=None, alpha=0.7, smooth=0.000001):

    if y_w is None:
        y_w = torch.ones_like(y_pred)
    tp = torch.sum(y_pred * y_t)
    fp = (1 - alpha) * torch.sum((y_pred * y_w) * (1 - y_t))
    fn = alpha * torch.sum(((1 - y_pred) * y_w) * y_t)

    numerator = tp
    denominator = tp + fp + fn

    score = (numerator + smooth) / (denominator + smooth)
    return 1 - score

def write_images(last_batch, writer, epoch, set_name='train'):
        b, _, h, w = last_batch.shape
        image = last_batch[:, [0], :, :]
        true_mask = last_batch[:, [1], :, :]
        pred_mask = last_batch[:, [2], :, :]

        indices = np.random.choice(b, min(b, 4), replace=False)
        cls_labels = {0: "background", 1: "trees"}
        mask_list = []
        for idx in indices:
            wb_masks = {
                "True_Mask": {"mask_data": true_mask[idx][0].cpu().numpy().astype(bool),
                              "class_labels": cls_labels},
                "Pred_Mask": {"mask_data": pred_mask[idx][0].cpu().numpy().astype(bool),
                              "class_labels": cls_labels}
            }

            image_ = image[idx][0].cpu().numpy()
            image_ = (image_ - image_.min()) / (image_.max() - image_.min() + 1e-6)
            image_ = np.stack([image_] * 3, axis=-1)
            image_ = (image_ * 255).astype(np.uint8)

            mask_list.append(wandb.Image(image_, masks=wb_masks))
        writer.log({
            f"{set_name}/predictions": mask_list, "epoch": epoch})


class Training:
    def __init__(self, **kwargs):
        # general parameters
        self.keyword = kwargs.get('keyword', 'test')

        # data parameters
        self.batch_size = kwargs.get('batch_size', 2)
        self.patch_size = kwargs.get('patch_size', 256)
        self.stretch_setting = kwargs.get('stretch_setting', 1)
        self.data_dir = kwargs.get('data_dir', None)

        # Training parameters
        self.epochs = kwargs.get('epochs', 300)
        self.num_workers = kwargs.get('num_workers', 8)
        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.boundary_weight = kwargs.get('boundary_weight', 1)
        self.model_size = kwargs.get('model_size', 'small')
        self.loss_function = kwargs.get('loss_function', 'tversky')
        self.alpha = kwargs.get('alpha', 0.5)
        self.runs_dir = kwargs.get('runs_dir', 'runs')
        self.load_model = kwargs.get('load', None)  # not using for now
        self.checkpoints_dir = os.path.join(self.runs_dir, self.keyword)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Visualisation
        self.use_wb = kwargs.get('use_wb', False)
        if self.use_wb:
            self.writer = wandb.init(
                project="Arctic",
                name=self.keyword,
                dir=self.runs_dir,
                config={k: v for k, v in kwargs.items() if k != 'load'}
            )

    def run_epoch(self,
                  model,
                  dataloader,
                  optimizer=None,
                  is_train=True):

        """Run one epoch of training or validation"""
        model.train() if is_train else model.eval()
        metrics_calc = utils.MetricsCalculator()
        total_loss = 0

        desc = "train" if is_train else "val"
        pbar = tqdm(dataloader, desc=desc)

        for batch in pbar:
            image, mask = batch[0].to(self.device), batch[1].to(self.device)

            # manually remove 47 pixel boundary from label mask
            mask = mask[:, :, 47:-47, 47:-47]  #fixme make it automatic finding border pixels

            if is_train:

                mask = (mask > 0.5).long()
                y_w = mask[:, [1]]
                y_w = torch.where(y_w==1, self.boundary_weight, 1)
                mask = mask[:, [0]]

                pred = model(image)

                # estimate loss function
                if self.loss_function == 'tversky':
                    prob = torch.sigmoid(pred)
                    loss = tversky(mask.float(), prob, y_w.float(), alpha=self.alpha)
                elif self.loss_function == 'ce':
                    loss = F.binary_cross_entropy_with_logits(pred, mask.float(), reduction="none")
                    loss = (loss * y_w).mean()
                else:
                    raise Exception("Unknown loss function")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                mask = mask[:, [0]]
                with torch.no_grad():
                    pred = model(image)
                    if self.loss_function == 'tversky':
                        prob = torch.sigmoid(pred)
                        loss = tversky(mask.float(), prob, alpha=self.alpha)
                    elif self.loss_function == 'ce':
                        loss = F.binary_cross_entropy_with_logits(pred, mask.float(), reduction="mean")
                    else:
                        raise Exception("Unknown loss function")

            total_loss += loss.item()
            metrics_calc.update((pred > 0).long(), mask)

            current = metrics_calc.compute()
            pbar.set_postfix(loss=total_loss/(pbar.n+1), iou=current['iou'], f1=current['f1'])

        results = metrics_calc.compute()
        results['loss'] = total_loss / len(dataloader)

        return results

    def save_checkpoint(self, name, model, optimizer, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'metrics': metrics
        }
        torch.save(checkpoint, os.path.join(self.checkpoints_dir, f'{name}.pth'))

    def train(self):
        # Setup data
        train_set = datagen.Datagen(self.data_dir,
                                    set_name="train",
                                    patch_size=self.patch_size,
                                    stretch_setting=self.stretch_setting)
        train_loader = DataLoader(train_set,
                                  self.batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=self.num_workers)

        val_set = datagen.Datagen(self.data_dir,
                                  set_name="val",
                                  patch_size=512)
        val_loader = DataLoader(val_set,
                                min(self.batch_size, 24),
                                drop_last=False,
                                shuffle=False,
                                num_workers=self.num_workers)

        # Setup model
        model = network.TinyUNet(size=self.model_size)
        model.apply(utils.init_kaiming)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_params)

        if self.load_model:
            model.load_state_dict(torch.load(self.load_model, map_location=self.device)["model"])

        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

        # Training loop
        print("Start training")
        best = {'precision': 0, 'recall': 0, 'f1': 0}

        # Estimate mean and std if normalize
        # mean_train, std_train = self.mean_nd_std()

        for epoch in range(self.epochs):
            train_results = self.run_epoch(model, train_loader, optimizer, is_train=True)
            val_results = self.run_epoch(model, val_loader, is_train=False)

            # Log to WandB
            log_data = {f'train_{k}': v for k, v in train_results.items()}
            log_data.update({f'val_{k}': v for k, v in val_results.items()})
            log_data['epoch'] = epoch + 1
            if self.use_wb:
                self.writer.log(log_data)

            # Save checkpoints
            all_metrics = {**{f'train_{k}': v for k, v in train_results.items()},
                          **{f'val_{k}': v for k, v in val_results.items()}}

            self.save_checkpoint('latest', model, optimizer, epoch, all_metrics)

            for metric in ['precision', 'recall', 'f1']:
                if val_results[metric] > best[metric]:
                    best[metric] = val_results[metric]
                    print(f"Best {metric}: {best[metric]:.5f} at epoch {epoch + 1}")
                    self.save_checkpoint(f'best_{metric}', model, None, epoch, all_metrics)

            # Log to CSV
            self.log_to_csv(epoch, train_results, val_results)

        self.write_checkpoint_summary()
        wandb.finish()

    def log_to_csv(self, epoch, train_results, val_results):
        """Write metrics to CSV file"""
        csv_path = os.path.join(self.checkpoints_dir, "metrics.csv")

        if epoch == 0:
            header = "epoch,train_loss,train_iou,train_precision,train_recall,train_f1,train_accuracy,"
            header += "train_tp,train_fp,train_fn,train_tn,val_loss,val_iou,val_precision,val_recall,"
            header += "val_f1,val_accuracy,val_tp,val_fp,val_fn,val_tn\n"
            with open(csv_path, "w") as f:
                f.write(header)

        row = f"{epoch+1},{train_results['loss']},{train_results['iou']},{train_results['precision']},"
        row += f"{train_results['recall']},{train_results['f1']},{train_results['accuracy']},"
        row += f"{train_results['tp']},{train_results['fp']},{train_results['fn']},{train_results['tn']},"
        row += f"{val_results['loss']},{val_results['iou']},{val_results['precision']},"
        row += f"{val_results['recall']},{val_results['f1']},{val_results['accuracy']},"
        row += f"{val_results['tp']},{val_results['fp']},{val_results['fn']},{val_results['tn']}\n"

        with open(csv_path, "a") as f:
            f.write(row)

    def write_checkpoint_summary(self):
        """Write metrics for all checkpointed models"""
        log_path = os.path.join(self.checkpoints_dir, 'checkpoint_summary.txt')

        with open(log_path, 'w') as f:
            f.write("=" * 80 + "\nCHECKPOINT MODELS SUMMARY\n" + "=" * 80 + "\n\n")

            for name in ['latest', 'best_precision', 'best_recall', 'best_f1']:
                checkpoint_path = os.path.join(self.checkpoints_dir, f'{name}.pth')

                if not os.path.exists(checkpoint_path):
                    f.write(f"Model: {name}.pth\nStatus: Not found\n" + "=" * 80 + "\n\n")
                    continue

                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                m = checkpoint.get('metrics', {})
                epoch = checkpoint.get('epoch', 'N/A')

                f.write(f"Model: {name}.pth\nEpoch: {epoch + 1 if isinstance(epoch, int) else epoch}\n")
                f.write("-" * 80 + "\n")

                for split in ['train', 'val']:
                    f.write(f"{split.capitalize()} Metrics:\n")
                    f.write(f"  Loss:      {m.get(f'{split}_loss', 0):.6f}\n")
                    f.write(f"  Precision: {m.get(f'{split}_precision', 0):.6f}\n")
                    f.write(f"  Recall:    {m.get(f'{split}_recall', 0):.6f}\n")
                    f.write(f"  F1 Score:  {m.get(f'{split}_f1', 0):.6f}\n")
                    f.write(f"  Accuracy:  {m.get(f'{split}_accuracy', 0):.6f}\n")
                    f.write(f"  IoU:       {m.get(f'{split}_iou', 0):.6f}\n")
                    f.write(f"  TP: {m.get(f'{split}_tp', 0)}, FP: {m.get(f'{split}_fp', 0)}, ")
                    f.write(f"FN: {m.get(f'{split}_fn', 0)}, TN: {m.get(f'{split}_tn', 0)}\n\n")

                f.write("=" * 80 + "\n\n")

        print(f"\nCheckpoint summary written to: {log_path}")


