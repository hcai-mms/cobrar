# %%
import pandas as pd
import numpy as np
import os
import tqdm
import shutil

# %%
df_interactions = pd.read_csv('../data/userid_trackid_timestamp.tsv', sep='\t')
print(df_interactions.shape)
df_interactions.head()

# %%
df_emma = pd.read_csv('../data/id_emma.tsv', sep='\t')
df_emma

# %%
df_interactions = df_interactions[df_interactions['track_id'].isin(df_emma.id)]
print(df_interactions.shape)
df_interactions.head()

# %%
# sparsity of dataset
n_users = df_interactions['user_id'].nunique()
n_items = df_interactions['track_id'].nunique()
n_interactions = df_interactions.shape[0]
sparsity = 1 - n_interactions / (n_users * n_items)
print(sparsity)

# %%
print(n_items)

# %%
print(n_users)

# %%
# export
df_interactions.to_csv('../data/userid_trackid_timestamp_emma.tsv', sep='\t', index=False)


