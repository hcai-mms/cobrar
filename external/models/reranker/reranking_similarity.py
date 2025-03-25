import pickle

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, haversine_distances, chi2_kernel, manhattan_distances
from sklearn.metrics import pairwise_distances




class Similarity(object):

    def __init__(self, data, similarity, k, modalitiy, multimodal_feature):
        self._data = data
        self._ratings = data.train_dict

        self._similarity = similarity
        self._k = k
        self.multimodal_feature = multimodal_feature
        self.modalitiy = modalitiy

        self._URM = self._data.sp_i_train

        self._users = self._data.users
        self._items = self._data.items
        self._private_users = self._data.private_users
        self._public_users = self._data.public_users
        self._private_items = self._data.private_items
        self._public_items = self._data.public_items

    def rerank_recommendations(self, recs):
        for u, user_recs in recs.items():
            user_id = self._public_users.get(u)
            user_recs_items = [rec for rec, _ in user_recs]
            user_recs_public_items = [self._public_items.get(rec) for rec in user_recs_items]
            user_items = self._URM[user_id].indices

            user_feature = np.mean(self.multimodal_feature[user_items], axis=0)
            user_recs_feature = self.multimodal_feature[user_recs_public_items]

            user_item_similarity = pairwise_distances(user_feature.reshape(1, -1), user_recs_feature, metric=self._similarity).flatten()

            # rerank and sort according to highest similarity
            recs[u][:self._k] = sorted(list(zip(user_recs_items[:self._k], user_item_similarity[:self._k])), key=lambda x: x[1], reverse=True)

        return recs