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

class CoBraR(DeepMF):
    r"""
        Deep Matrix Factorization Models for Recommender Systems.

        For further details, please refer to the `paper <https://www.ijcai.org/Proceedings/2017/0447.pdf>`_

        Args:
            lr: Learning rate
            reg: Regularization coefficient
            embedding_dim: shared embedding dimension
            user_mlp: List of units for each layer
            item_mlp: List of activation functions
            similarity: Number of factors dimension


        To include the recommendation model, add it to the config file adopting the following pattern:

        .. code:: yaml

          models:
            DMF:
              meta:
                save_recs: True
              epochs: 10
              batch_size: 512
              lr: 0.0001
              reg: 0.001
              emb_dim: 32
              user_mlp: (64,32)
              item_mlp: (64,32)
              similarity: cosine
        """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        DeepMF.__init__(self, data, config, params, *args, **kwargs)
        self._params_list += [
            ("_collaborative_branch", "collaborative_branch", "cb", [1024, 512], list, None),
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
            random_seed=self._seed
        )

    @property
    def name(self):
        return "CoBraR"\
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"