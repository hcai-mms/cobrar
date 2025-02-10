from tqdm import tqdm
import numpy as np
import torch
import os
import math

from elliot.utils.write import store_recommendation
from elliot.dataset.samplers import pointwise_pos_neg_sampler as pws
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .FactorizationMachineModel import FactorizationMachineModel


class FactorizationMachine(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.001, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_dropout_rate", "dropout_rate", "dropout_rate", 0.0, float, None),
            ("_batch_size", "batch_size", "batch_size", 512, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_loader", "loader", "load", "ItemAttributes", None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        if (hasattr(self._side, "nfeatures")) and (hasattr(self._side, "feature_map")):
            self._nfeatures = self._side.nfeatures
            self._item_array = self.get_item_fragment()
        else:
            self._nfeatures = 0

        self._field_dims = [self._num_users, self._num_items, self._nfeatures]

        self._sampler = pws.Sampler(self._data.i_train_dict)

        self._model = FactorizationMachineModel(
            self._num_users,
            self._num_items,
            self._nfeatures,
            embed_dim=self._factors,
            learning_rate=self._learning_rate,
            random_seed=self._seed,
        )

    @property
    def name(self):
        return "FM" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    batch = self.prepare_fm_transaction(batch)
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        user_batch_size = 10 if self._num_items > 1000 else self._batch_size # small batch size needed for memory
        for index, offset in enumerate(range(0, self._num_users, user_batch_size)):
            offset_stop = min(offset + user_batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop, self._item_array)
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

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False

    def get_item_fragment(self):
        if not self._nfeatures:
            return []

        item_features_list = []
        padding_value = 0

        for item in range(self._num_items):
            i_features = self._side.feature_map[item]

            # Pad the list to the maximum feature length
            padded_features = i_features + [padding_value] * (self._nfeatures - len(i_features))
            item_features_list.append(padded_features)

        return np.array(item_features_list, dtype=np.int32)

    def prepare_fm_transaction(self, batch):
        return batch[0], batch[1], self._item_array[batch[1]], batch[2]
