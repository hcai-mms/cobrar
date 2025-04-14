import ntpath
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from elliot.recommender import ProxyRecommender
from elliot.recommender.base_recommender_model import init_charger

import wandb

from .reranking_similarity import Similarity


class RerankingRecommender(ProxyRecommender):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a reranking recommender based on emotion to rerank already generated recommendations.
        :param name: data loader object
        :param path: path to the directory rec. results
        :param args: parameters
        """
        self._random = np.random

        self._params_list = [
            ("_input_path", "input_path", "input_path", "", None, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_k", "k", "k", 10, None, None),
            ("_modalitiy", "modalitiy", "modality", "emotion", None, None),
            ("_loader", "loader", "load", "EmotionAttribute", None, None)
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self.__setattr__(f'''_side_{self._modalitiy}''',
                         self._data.side_information.__getattribute__(f'''{self._loader}'''))

        multimodal_feature = self.__getattribute__(f'''_side_{self._modalitiy}''').object.get_all_features()

        self._reranking_model = Similarity(data=self._data,
                                 similarity=self._similarity,
                                 k=self._k,
                                 modalitiy=self._modalitiy,
                                 multimodal_feature=multimodal_feature)

        self._base_model_name = ntpath.basename(self._input_path).rsplit(".",1)[0].split('_')[0]

        wandb.init(
            project=f"Reranker{self._base_model_name}-{os.path.basename(os.path.dirname(config.data_config.dataset_path))}",
            name=self.name,
            config={
                **{
                    "similarity": self._similarity,
                    "k": self._k,
                    "modalitiy": self._modalitiy,
                    "path": ntpath.basename(self._input_path)
                }
            },
            reinit=True,
        )

    @property
    def name(self):
        return f"Reranker{self._base_model_name}_{self._similarity}_{self._k}_{self._modalitiy}"

    def train(self):
        print("Reading recommendations")
        recs = self.read_recommendations(self._input_path)

        self._recommendations = self._reranking_model.rerank_recommendations(recs)

        print("Evaluating recommendations")
        self.evaluate(wandb_log=True)

    def get_recommendations(self, top_k):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(top_k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k):
        candidate_items = {}
        setItem = set(range(mask.shape[1]))
        for user in tqdm(range(mask.shape[0])):
            itemFalse = set(np.where(mask[user, :] == False)[0].tolist())
            itemTrue = list(setItem.difference(itemFalse))
            candidate_items[self._data.private_users[user]] = [self._data.private_items[item] for item in itemTrue]

        recs = {}
        for u, user_recs in self._recommendations.items():
            user_cleaned_recs = []
            user_candidate_items = set(candidate_items[u])
            for p, (item, prediction) in enumerate(user_recs):
                if p >= k:
                    break
                if item in user_candidate_items:
                    user_cleaned_recs.append((item, prediction))
            recs[u] = user_cleaned_recs
        return recs