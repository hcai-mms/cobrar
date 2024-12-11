# %%
import pandas as pd
import numpy as np
import os
import shutil

# %%
df_interactions = pd.read_csv('../data/userid_trackid_timestamp.tsv', sep='\t')
print(df_interactions.shape)

# %%
df_interactions['period'] = pd.to_datetime(df_interactions['timestamp']).dt.to_period('Y')
df_interactions = df_interactions[df_interactions['period'] == '2018']
print(df_interactions.shape)

# %%
# binarization: we convert the interactions to binary implicit feedback with a threshold of 2 on the interaction counts (reducing false-positive interactions)
def filter_interactions(listening_history: pd.DataFrame, min_interactions: int = 2, verbose: bool = True):
    lhs_count = listening_history.value_counts(subset=['user_id', 'track_id'])
    lhs_count = lhs_count[lhs_count >= min_interactions]
    listening_history = listening_history.set_index(['user_id', 'track_id']).loc[lhs_count.index]
    listening_history = listening_history.reset_index()
    return listening_history

df_interactions = filter_interactions(df_interactions, min_interactions=2)
print(df_interactions.shape)
df_interactions.head()

# %%
# sparsity of dataset
n_users = df_interactions['user_id'].nunique()
n_items = df_interactions['track_id'].nunique()
n_interactions = df_interactions.shape[0]
sparsity = 1 - n_interactions / (n_users * n_items)
sparsity

# %%
n_items

# %%
n_users

# %%
# export
df_interactions.to_csv('../data/userid_trackid_timestamp_onion.tsv', sep='\t', index=False)


