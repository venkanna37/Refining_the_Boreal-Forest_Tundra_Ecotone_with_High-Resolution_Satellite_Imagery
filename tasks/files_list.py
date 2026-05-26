import glob
import os

root_dir = "/scratch/project_465002698/venky/projects/arctic/all_mosaics/"

files = [
    f for f in glob.glob(f"{root_dir}/**/*.tif", recursive=True)
    if os.path.dirname(f) != root_dir and 'browse' not in f
]

import ipdb; ipdb.set_trace()
for f in files:
    print(f)