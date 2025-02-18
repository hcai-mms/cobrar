import numpy as np
from ast import literal_eval as make_tuple
from tqdm import tqdm
import torch
import os

from elliot.utils.write import store_recommendation
from elliot.dataset.samplers import pointwise_pos_neg_ratio_ratings_sampler as pws
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

# from elliot.dataset.samplers import pointwise_pos_neg_ratio_ratings_sampler as pws
# from elliot.recommender.recommender_utils_mixin import RecMixin
# from elliot.recommender.base_recommender_model import BaseRecommenderModel
# from elliot.recommender.base_recommender_model import init_charger

from .DeepMFModel import DeepMFModel

class DeepMF(RecMixin, BaseRecommenderModel):
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
        self._params_list = [
            ("_embedding_dim", "embedding_dim", "emb_dim", 32, None, None),
            ("_user_mlp", "user_mlp", "umlp", [64, 32], list, None),
            ("_item_mlp", "item_mlp", "imlp", [64, 32], list, None),
            # ToDo check
            ("_neg_ratio", "neg_ratio", "negratio", 5, None, None),
            ("_reg", "reg", "reg", 0.001, None, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_learning_rate", "lr", "lr", 0.0001, None, None),
            ("_mu", "mu", "mu", 0.0001, None, None),
        ]
        self.autoset_params()

        self._max_ratings = np.max(self._data.sp_i_train_ratings)
        self._transactions_per_epoch = self._data.transactions + self._neg_ratio * self._data.transactions

        if self._batch_size < 1:
            self._batch_size = self._data.transactions + self._neg_ratio * self._data.transactions

        self._sampler = pws.Sampler(self._data.i_train_dict, self._data.sp_i_train_ratings, self._neg_ratio)

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = DeepMFModel(
            num_users=self._num_users,
            num_items=self._num_items,
            embedding_dim=self._embedding_dim,
            user_mlp=self._user_mlp,
            item_mlp=self._item_mlp,
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
        return "DeepMF"\
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._transactions_per_epoch // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._transactions_per_epoch, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        self._model.eval()
        with torch.no_grad():
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(offset, offset_stop)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)

                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")
