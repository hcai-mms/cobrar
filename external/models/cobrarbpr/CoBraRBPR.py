import numpy as np
from ast import literal_eval as make_tuple
from tqdm import tqdm
import torch
import os
import math


from .custom_sampler import Sampler
from elliot.recommender.base_recommender_model import init_charger

from ..deepmf.DeepMF import DeepMF
from .CoBraRBPRModel import CoBraRBPRModel

import wandb

class CoBraRBPR(DeepMF):
    r"""
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        DeepMF.__init__(self, data, config, params, *args, **kwargs)
        self._params_list += [
            ("_collaborative_branch", "collaborative_branch", "cb", [1024, 512], list, None),
            # Todo add to config
            ("_num_neg", "num_neg", "ng", 5, int, None),
        ]
        self.autoset_params()
        self._sampler = Sampler(self._data.i_train_dict, self._num_neg, self._seed)

        self._model = CoBraRBPRModel(
            num_users=self._num_users,
            num_items=self._num_items,
            embedding_dim=self._embedding_dim,
            user_mlp=self._user_mlp,
            item_mlp=self._item_mlp,
            collaborative_branch=self._collaborative_branch,
            reg=self._reg, # ToDo check if needed
            similarity=self._similarity,
            max_ratings=self._max_ratings, # ToDo check if needed
            sp_i_train_ratings=self._data.sp_i_train_ratings,
            learning_rate=self._learning_rate,
            mu=self._mu, # ToDo check if needed
            random_seed=self._seed
        )
        wandb.init(
            project=f"CoBraRBPR-{config.data_config.dataset_path.split('/')[-2]}",
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
                },
                **{f"user_layer-{ii}": layer for ii, layer in enumerate(self._user_mlp)},
                **{f"item_layer-{ii}": layer for ii, layer in enumerate(self._item_mlp)},
                **{f"collaborative_branch-{ii}": layer for ii, layer in enumerate(self._collaborative_branch)},
            },
            reinit=True,
        )

    @property
    def name(self):
        return "CoBraRBPR"\
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"
    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            n_batch = int(self._data.transactions / self._batch_size) if self._data.transactions % self._batch_size == 0 else int(self._data.transactions / self._batch_size) + 1
            self._data.edge_index = self._data.edge_index.sample(frac=1, replace=False).reset_index(drop=True)
            # edge_index = np.array([self._data.edge_index['userId'].tolist(), self._data.edge_index['itemId'].tolist()])
            with tqdm(total=n_batch, disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    self._model.optimizer.zero_grad()
                    current_loss = self._model.train_step(batch)
                    loss += current_loss

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()
                # self._model.lr_scheduler.step()

            self.evaluate(it, loss / (it + 1))
    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        item_repr = self._model.get_item_representations(torch.arange(self._num_items).to(self._model.device))
        # print(item_repr.shape)

        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            user_indices = torch.arange(index * self._batch_size, offset_stop).to(self._model.device)

            user_repr = self._model.get_user_representations(user_indices)
            # print(user_repr)

            predictions = self._model.predict(user_repr, item_repr)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test