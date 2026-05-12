import torch
import rasterio
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from pipelines import network



@torch.no_grad()
def ensemble_union_predict(models, input_patch, mode = "majority"):
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
    # MAJORITY
    if mode == "majority":
        mask = torch.mean(stacked, dim=0)
        mask = (mask > 0.5).to(torch.uint8)
    elif mode == "intersection":
        # INTERSECTION
        mask = stacked > 0.5
        mask = torch.all(mask, dim=0).to(torch.uint8)

    return mask


def apply_model_on_geotiff(
        geotiff_path,
        checkpoint_path,
        device=None,
        rescale_value = 2000
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}

    for i, path in enumerate(checkpoint_path):
        model = network.TinyUNet()
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

    with torch.no_grad():
        patch = image.to(device)
        pred = ensemble_union_predict(models, patch)[0, 0]

    return pred.cpu().numpy()


def common_prefix(name):
    name = name.replace("_test_labels", "").replace("_labels", "")
    parts = name.split("_")
    return "_".join(parts)  #[:3])


def pixel_metrics_table(labels_dir,
                        predictions_dir,
                        out_csv=None
                        ):
    rows = []

    label_files = sorted(Path(labels_dir).glob("*.tif"))
    pred_files = list(Path(predictions_dir).glob("*.tif"))
    print(len(label_files), len(pred_files))

    for label_path in label_files:
        prefix = common_prefix(label_path.stem)

        matches = [
            p for p in pred_files
            if p.stem.startswith(prefix)
        ]

        if len(matches) != 1:
            continue

        pred_path = matches[0]

        with rasterio.open(label_path) as src:
            y_true = src.read(1)
            y_true[y_true > 1] = 0
            y_true = y_true.astype(bool)

        with rasterio.open(pred_path) as src:
            y_pred = src.read(1).astype(bool)

        y_true = y_true[47:-47, 47:-47]
        y_pred = y_pred[47:-47, 47:-47]
        # print(y_true.sum())
        # print(y_pred.sum())
        tp = np.logical_and(y_pred, y_true).sum()
        tn = np.logical_and(~y_pred, ~y_true).sum()
        fp = np.logical_and(y_pred, ~y_true).sum()
        fn = np.logical_and(~y_pred, y_true).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        parts = prefix.split("_")
        location = parts[0]
        season = parts[2]

        rows.append({
            "Location": location,
            "Season": season,
            "True Positives": int(tp),
            "True Negatives": int(tn),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy
        })

    df = pd.DataFrame(rows)

    if df.empty:
        if out_csv is not None:
            out_csv = Path(out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
        return df

    tp = df["True Positives"].sum()
    tn = df["True Negatives"].sum()
    fp = df["False Positives"].sum()
    fn = df["False Negatives"].sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    avg_row = {
        "Location": "ALL",
        "Season": "ALL",
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "Precision": precision ,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy
    }

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    print(df)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--keyword", type=str, help="Keyword used to save the model")
    parser.add_argument("--labels_dir", type=str, help="labels dir", default="data/testdata/labels")
    params = vars(parser.parse_args())
    # predictions_dir = f"./data/testdata/pred_{params.get('keyword')}"
    predictions_dir = f"data/predictions/pred_{params.get('keyword')}"
    # out_csv = f"./data/testdata/results_{params.get('keyword')}.csv"
    out_csv = f"data/predictions/results_{params.get('keyword')}.csv"
    pixel_metrics_table(params.get("labels_dir"), predictions_dir, out_csv)