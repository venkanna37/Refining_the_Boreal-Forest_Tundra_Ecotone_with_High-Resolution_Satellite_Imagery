"""
Creating script for five folds of data
"""

import os
import shutil
import geopandas as gpd
from sklearn.model_selection import KFold

# ------------------------------------------------------------------
# Five-fold split
# ------------------------------------------------------------------
base_directory = '/home/venky/Documents/projects/simple_segment/data/geofolds2'
foldernames = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
df = gpd.read_file("/home/venky/Documents/projects/simple_segment/data/dataset9010/train_bboxes.geojson")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
geoids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# import ipdb; ipdb.set_trace()
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(geoids)):
    print(fold_idx, train_idx, val_idx)
    fold_name = foldernames[fold_idx]
    fold_path = os.path.join(base_directory, fold_name)

    # Create directory structure
    for split in ['train', 'val']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(fold_path, split, sub), exist_ok=True)

    # --------------------------------------------------------------
    # Copy training files
    # --------------------------------------------------------------
    for i in train_idx:
        i = i+1
        folddf = df[df['area_id']==i]
        for i, row in folddf.iterrows():
            src_ip = os.path.join('/home/venky/Documents/projects/simple_segment/data/dataset9010/train/images/', row['image'])
            label_file = row['image'][:-5]+'annotation_and_boundary_0.tif'
            src_lp = os.path.join('/home/venky/Documents/projects/simple_segment/data/dataset9010/train/labels/', label_file)
            if not os.path.exists(src_ip):
                src_ip = os.path.join('/home/venky/Documents/projects/simple_segment/data/dataset9010/val/images/',
                                      row['image'])
                src_lp = os.path.join('/home/venky/Documents/projects/simple_segment/data/dataset9010/val/labels/',
                                      label_file)

            shutil.copy(
                src_ip,
                os.path.join(fold_path, 'train', 'images', os.path.basename(src_ip))
            )
            shutil.copy(
                src_lp,
                os.path.join(fold_path, 'train', 'labels', os.path.basename(src_lp))
            )

    # --------------------------------------------------------------
    # Copy validation files
    # --------------------------------------------------------------
    for i in val_idx:
        i = i + 1
        folddf = df[df['area_id']==i]
        for i, row in folddf.iterrows():
            src_ip = os.path.join('/home/venky/Documents/projects/simple_segment/data/dataset9010/train/images/', row['image'])
            label_file = row['image'][:-5]+'annotation_and_boundary_0.tif'
            src_lp = os.path.join('/home/venky/Documents/projects/simple_segment/data/dataset9010/train/labels/', label_file)
            if not os.path.exists(src_ip):
                src_ip = os.path.join('/home/venky/Documents/projects/simple_segment/data/dataset9010/val/images/',
                                      row['image'])
                src_lp = os.path.join('/home/venky/Documents/projects/simple_segment/data/dataset9010/val/labels/',
                                      label_file)

            shutil.copy(
                src_ip,
                os.path.join(fold_path, 'val', 'images', os.path.basename(src_ip))
            )
            shutil.copy(
                src_lp,
                os.path.join(fold_path, 'val', 'labels', os.path.basename(src_lp))
            )

    print(f"{fold_name}: "
          f"{len(train_idx)} train samples, "
          f"{len(val_idx)} val samples")

print("Five-fold dataset creation completed.")

