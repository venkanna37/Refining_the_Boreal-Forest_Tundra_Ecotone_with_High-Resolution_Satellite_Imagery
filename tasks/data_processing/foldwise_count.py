import geopandas as gpd
import os
import pandas as pd

data_dir = '/home/venky/Documents/projects/arctic/data/new_location/'

prev_df = gpd.read_file(os.path.join(data_dir, 'train_bboxes.geojson'))
prev_df['area_id'] = prev_df['area_id'] -1

new_df = gpd.read_file(os.path.join(data_dir, 'new_location_bbox.geojson'))
new_df['area_id'] = 16
import ipdb; ipdb.set_trace()

# combine both prev_df and new_df
df_merged = pd.concat([prev_df, new_df], ignore_index=True, sort=False)
df_merged = gpd.GeoDataFrame(df_merged, geometry='geometry', crs=prev_df.crs)
output_geojson = os.path.join(data_dir, "all_bbox.geojson")
df_merged.to_file(output_geojson, driver="GeoJSON")
import ipdb; ipdb.set_trace()


def count_train_and_val_patches(df, val_ids):
    df = df.groupby('org_id').image.count().reset_index()
    df.rename(columns={'image': 'count'}, inplace=True)
    val_df = df[df['org_id'].isin(val_ids)]
    train_df = df[~df['org_id'].isin(val_ids)]
    val_count = val_df['count'].sum()
    train_count = train_df['count'].sum()
    return train_count, val_count

fold1_ids = [0, 1, 5, 14]
fold2_ids = [8, 11, 13]
fold3_ids = [2, 9, 15]
fold4_ids = [4, 7, 10]
fold5_ids = [3, 6, 12]

# fold1 counts
t_count, val_count = count_train_and_val_patches(prev_df, fold1_ids)
print(f"Fold1:- train counts: {t_count}, val counts: {val_count}")

# fold2 counts
t_count, val_count = count_train_and_val_patches(prev_df, fold2_ids)
print(f"Fold2:- train counts: {t_count}, val counts: {val_count}")

t_count, val_count = count_train_and_val_patches(prev_df, fold3_ids)
print(f"Fold3:- train counts: {t_count}, val counts: {val_count}")

t_count, val_count = count_train_and_val_patches(prev_df, fold4_ids)
print(f"Fold4:- train counts: {t_count}, val counts: {val_count}")

t_count, val_count = count_train_and_val_patches(prev_df, fold5_ids)
print(f"Fold5:- train counts: {t_count}, val counts: {val_count}")