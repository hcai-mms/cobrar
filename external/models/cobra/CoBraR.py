import numpy as np
from ast import literal_eval as make_tuple
from tqdm import tqdm
import torch
import os

from elliot.utils.write import store_recommendation
from elliot.dataset.samplers import pointwise_pos_neg_ratio_ratings_sampler as pws
from elliot.recommender.base_recommender_model import init_charger

from ..deepmf.DeepMF import DeepMF
from .CoBraRModel import CoBraRModel

import wandb

class CoBraR(DeepMF):
    r"""
        Collaborative Branch for Recommendation. Under review for RecSys'25
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        DeepMF.__init__(self, data, config, params, *args, **kwargs)
        self._params_list += [
            ("_collaborative_branch", "collaborative_branch", "cb", [1024, 512], list, None),
            ("_dropout", "dropout", "do", -1., float, None),
            ("_batch_norm", "batch_norm", "bn", False, bool, None),
        ]
        self.autoset_params()

        self._model = CoBraRModel(
            num_users=self._num_users,
            num_items=self._num_items,
            embedding_dim=self._embedding_dim,
            user_mlp=self._user_mlp,
            item_mlp=self._item_mlp,
            collaborative_branch=self._collaborative_branch,
            reg=self._reg,
            similarity=self._similarity,
            max_ratings=self._max_ratings,
            sp_i_train_ratings=self._data.sp_i_train_ratings,
            learning_rate=self._learning_rate,
            mu=self._mu,
            dropout=self._dropout,
            batch_norm=self._batch_norm,
            random_seed=self._seed
        )
        wandb.init(
            project=f"CoBraR-{config.data_config.dataset_path.split('/')[-2]}",
            name=self.name,
            config={
                **{
                    "learning_rate": self._learning_rate,
                    "factors": self._embedding_dim,
                    "reg": self._reg,
                    "similarity": self._similarity,
                    "max_ratings": self._max_ratings,
                    "batch_size": self._batch_size,
                    "neg_ratio": self._neg_ratio,
                    "mu": self._mu,
                    "dropout": self._dropout,
                    "batch_norm": self._batch_norm,
                },
                **{f"user_layer-{ii}": layer for ii, layer in enumerate(self._user_mlp)},
                **{f"item_layer-{ii}": layer for ii, layer in enumerate(self._item_mlp)},
                **{f"collaborative_branch-{ii}": layer for ii, layer in enumerate(self._collaborative_branch)},
            },
            reinit=True,
        )

    @property
    def name(self):
        return "CoBraR"\
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"