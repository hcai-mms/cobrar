import pandas as pd 
import numpy as np

def filter_interactions_binary(listening_history: pd.DataFrame, user_col='user_id', min_interactions: int = 2, verbose: bool = True):
    lhs_count = listening_history.value_counts(subset=[user_col, 'track_id'])
    lhs_count = lhs_count[lhs_count >= min_interactions]
    listening_history = listening_history.set_index([user_col, 'track_id']).loc[lhs_count.index]
    listening_history = listening_history.reset_index()
    return listening_history

def filter_interactions_interval(listening_history: pd.DataFrame, user_col='user_id', min_interaction_interval=30):
    listening_history = listening_history.sort_values(by=[user_col, 'timestamp'])
    listening_history['timestamp'] = pd.to_datetime(listening_history['timestamp'])
    listening_history['time_diff'] = listening_history.groupby(user_col)['timestamp'].diff().dt.total_seconds().fillna(0)
    listening_history = listening_history[(listening_history['time_diff'] >= min_interaction_interval) | 
                                         (listening_history['time_diff'] == 0)]
    listening_history = listening_history.drop(columns=['time_diff'])
    return listening_history

def delete_duplicates(listening_history: pd.DataFrame, user_col='user_id'):
    initial_shape = listening_history.shape
    listening_history = listening_history.drop_duplicates(subset=[user_col, 'track_id'], keep='first')
    #print(f"Deleted {initial_shape[0] - listening_history.shape[0]} duplicates")
    return listening_history

def k_core_filtering(df, user_col='user_id', item_col='track_id', k_user=5, k_item=5, max_iterations=50):
    filtered_df = df.copy()
    
    iteration = 0
    users_removed = 0
    items_removed = 0
    prev_shape = (-1, -1)
    current_shape = filtered_df.shape
    
    # Iterate until convergence or max iterations
    while (prev_shape != current_shape) and (iteration < max_iterations):
        iteration += 1
        prev_shape = current_shape
        
        # Count interactions for each user and item
        user_counts = filtered_df[user_col].value_counts()
        item_counts = filtered_df[item_col].value_counts()
        
        # Find users and items that don't meet the threshold
        users_to_keep = user_counts[user_counts >= k_user].index
        items_to_keep = item_counts[item_counts >= k_item].index
        
        # Filter the dataframe
        users_removed_this_iter = filtered_df[~filtered_df[user_col].isin(users_to_keep)][user_col].nunique()
        items_removed_this_iter = filtered_df[~filtered_df[item_col].isin(items_to_keep)][item_col].nunique()
        
        users_removed += users_removed_this_iter
        items_removed += items_removed_this_iter
        
        # Apply the filter
        filtered_df = filtered_df[
            filtered_df[user_col].isin(users_to_keep) & 
            filtered_df[item_col].isin(items_to_keep)
        ]
        
        current_shape = filtered_df.shape
    
    return filtered_df

def create_sessions(listening_history: pd.DataFrame, session_threshold: int = 60 * 60 * 24, verbose: bool = True):
    listening_history['timestamp'] = pd.to_datetime(listening_history['timestamp'])
    listening_history.sort_values(by=['user_id', 'timestamp'], inplace=True)
    cond1 = listening_history.timestamp - listening_history.timestamp.shift(1) > pd.Timedelta(30, 'm')
    cond2 = listening_history.user_id != listening_history.user_id.shift(1)
    listening_history['session_id'] = (cond1 | cond2).cumsum()
    return listening_history