import pandas as pd
import numpy as np
import torch

df = pd.read_csv("./nfe/experiments/gru_ode_bayes/data_preproc/full_dataset.csv")
# print(df[df["ID"] == 6])
print(df.head())
df = df.reset_index()
df = df.rename(columns={'HADM_ID':'ID', 'TIME_STAMP':'Time'})
df["Time"] = df['Time'].map(lambda x: pd.to_timedelta(x).seconds)
df.loc[:, 'Time'] = df['Time'] / 1000

print(df.head())
print(df.shape)

T_val = 2.160
max_val_samples = 3
before_idx = df.loc[df['Time'] <= T_val, 'ID'].unique()
after_idx = df.loc[df['Time'] > T_val, 'ID'].unique()

valid_idx = np.intersect1d(before_idx, after_idx)
df = df.loc[df['ID'].isin(valid_idx)].copy()

df_before = df.loc[df['Time'] <= T_val].copy()
df_after = df.loc[df['Time'] > T_val].sort_values('Time').copy()

df_after = df_after.groupby('ID').head(max_val_samples).copy()
df = df_before  # We remove observations after T_val
df_after.ID = df_after.ID.astype(np.int)
# df_after.sort_values('Time', inplace=True)

print(df_after.head(174))
print(df_after.shape)

# label_df = label_df.loc[label_df['ID'].isin(valid_idx)].copy()
# init_cov_df = init_cov_df.loc[init_cov_df['ID'].isin(valid_idx)].copy()