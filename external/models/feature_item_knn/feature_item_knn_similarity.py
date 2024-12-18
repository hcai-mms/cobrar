import pickle

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, haversine_distances, chi2_kernel, manhattan_distances
from sklearn.metrics import pairwise_distances




class Similarity(object):
    """
    Simple kNN class
    """

    def __init__(self, data, attribute_matrix, num_neighbors, similarity, implicit):
        self._data = data
        self._ratings = data.train_dict
        self._attribute_matrix = attribute_matrix
        self._num_neighbors = num_neighbors
        self._similarity = similarity
        self._implicit = implicit

        if self._implicit:
            self._URM = self._data.sp_i_train
        else:
            self._URM = self._data.sp_i_train_ratings

        self._users = self._data.users
        self._items = self._data.items
        self._private_users = self._data.private_users
        self._public_users = self._data.public_users
        self._private_items = self._data.private_items
        self._public_items = self._data.public_items

    def initialize(self):
        """
        This function initialize the data model
        """

        supported_similarities = ["cosine", "dot", ]
        supported_dissimilarities = ["euclidean", "manhattan", "haversine",  "chi2", 'cityblock', 'l1', 'l2', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        print(f"\nSupported Similarities: {supported_similarities}")
        print(f"Supported Distances/Dissimilarities: {supported_dissimilarities}\n")

        self._similarity_matrix = np.empty((len(self._items), len(self._items)))

        self.process_similarity(self._similarity)
        ##############
        data, rows_indices, cols_indptr = [], [], []

        column_row_index = np.arange(len(self._data.items), dtype=np.int32)

        for item_idx in range(len(self._data.items)):
            cols_indptr.append(len(data))
            column_data = self._similarity_matrix[:, item_idx]

            non_zero_data = column_data != 0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-self._num_neighbors:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

        cols_indptr.append(len(data))

        W_sparse = sparse.csc_matrix((data, rows_indices, cols_indptr),
                                     shape=(len(self._data.items), len(self._data.items)), dtype=np.float32).tocsr()
        self._preds = self._URM.dot(W_sparse).toarray()
        ##############
        # self.compute_neighbors()

        del self._similarity_matrix

    def process_similarity(self, similarity):
        if similarity == "cosine":
            self._similarity_matrix = cosine_similarity(self._attribute_matrix)
        elif similarity == "dot":
            self._similarity_matrix = (self._attribute_matrix @ self._attribute_matrix.T).toarray()
        elif similarity == "euclidean":
            self._similarity_matrix = (1 / (1 + euclidean_distances(self._attribute_matrix)))
        elif similarity == "manhattan":
            self._similarity_matrix = (1 / (1 + manhattan_distances(self._attribute_matrix)))
        elif similarity == "haversine":
            self._similarity_matrix = (1 / (1 + haversine_distances(self._attribute_matrix)))
        elif similarity == "chi2":
            self._similarity_matrix = (1 / (1 + chi2_kernel(self._attribute_matrix)))
        elif similarity in ['cityblock', 'l1', 'l2']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._attribute_matrix, metric=similarity)))
        elif similarity in ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._attribute_matrix.toarray(), metric=similarity)))
        else:
            raise Exception("Not implemented similarity")

    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self._preds[user_id]
        # user_items = self._ratings[u].keys()
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(user_recs)])

        # indices, values = zip(*predictions.items())
        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    # @staticmethod
    # def score_item(neighs, user_items):
    #     num = sum([v for k, v in neighs.items() if k in user_items])
    #     den = sum(np.power(list(neighs.values()), 1))
    #     return num/den if den != 0 else 0

    def get_model_state(self):
        saving_dict = {}
        saving_dict['_preds'] = self._preds
        saving_dict['_similarity'] = self._similarity
        saving_dict['_num_neighbors'] = self._num_neighbors
        saving_dict['_implicit'] = self._implicit
        return saving_dict

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._similarity = saving_dict['_similarity']
        self._num_neighbors = saving_dict['_num_neighbors']
        self._implicit = saving_dict['_implicit']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
