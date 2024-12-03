# %%
import pandas as pd
import numpy as np
import os
import tqdm
import shutil

# %%
sample = False
sample_size = 3e5

dataset = 'm4a' # m4a or emma
input_data_dir = f'../data/'

features = {
    'emotion_embeddings': ['id_gems.tsv'],
    'audio_embeddings': ['id_maest.tsv'],
    'textual_embeddings': ['id_bert.tsv'],
    'visual_embeddings': ['id_resnet.tsv']
}

if sample:
    output_dir = f'../dataset/{dataset}_sample/'
else:
    output_dir = f'../dataset/{dataset}/'

os.makedirs(output_dir, exist_ok=True)

# %%
if sample:
    df_interactions = pd.read_csv(f'../data/userid_trackid_timestamp_{dataset}.tsv', sep='\t', nrows=sample_size)
else:
    df_interactions = pd.read_csv(f'../data/userid_trackid_timestamp_{dataset}.tsv', sep='\t')
    
df_interactions['rating'] = 5
sample_items = df_interactions['track_id'].unique()

# map item and users to numbers
df_interactions['user_id_int'] = df_interactions['user_id'].astype('category').cat.codes
df_interactions['track_id_int'] = df_interactions['track_id'].astype('category').cat.codes

#df_interactions[['user_id', 'track_id', 'rating', 'timestamp']].to_csv(os.path.join(output_dir, 'interactions.tsv'), index=False, sep='\t', header=False)
df_interactions[['user_id_int', 'track_id_int', 'rating', 'timestamp']].to_csv(os.path.join(output_dir, 'interactions.tsv'), index=False, sep='\t', header=False)


print(df_interactions.shape)
df_interactions.head()

# %%
item_id_map = df_interactions[['track_id', 'track_id_int']].drop_duplicates()
item_id_map.set_index('track_id', inplace=True)
item_id_map.head()

# %%
del df_interactions

# %%
for feature_type in features:
    for file in features[feature_type]:
        print('Processing', file)
        df = pd.read_csv(os.path.join(input_data_dir, file), index_col=0, sep='\t')
        df = df.merge(item_id_map, left_index=True, right_index=True)
        feature_name = file.split('.')[0].split('_')[1]
        if sample:
            df = df.reindex(sample_items).dropna()
            
        folder_path = os.path.join(output_dir, feature_type, feature_name)

        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)

        for index, row in tqdm.tqdm(df.iterrows()):
            #np.save(os.path.join(folder_path, str(index) + '.npy'), row.values)
            np.save(os.path.join(folder_path, str(int(row['track_id_int'])) + '.npy'), row.values[:-1])

# %%



