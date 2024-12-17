import torch
import os
import numpy as np
from tqdm import tqdm
from ast import literal_eval as make_tuple

from elliot.utils.write import store_recommendation

from elliot.recommender import BaseRecommenderModel
from .BiVAEModel import BiVAECFModel
from .custom_sampler import Sampler
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger


class BiVAE(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_num_neg", "num_neg", "num_neg", 128, int, None),
            ("_num_sample", "num_sample", "num_sample", 0.5, float, None),
            ("_temperature", "temperature", "temperature", 1.0, float, None),
            ("_reg_weight", "reg_weight", "reg_weight", 0.01, float, None),
            ("_lr_lambda", "lr_lambda", "lr_lambda", 0.5, float, None),
            ("_multimod_factors", "multimod_factors", "multimod_factors", 64, int, None),
            ("_modalities", "modalities", "modalites", "('visual','textual')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_loaders", "loaders", "loads", "('VisualAttribute','TextualAttribute')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-"))
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._sampler = Sampler(self._data.i_train_dict, self._num_neg, self._seed)

        for m_id, m in enumerate(self._modalities):
            self.__setattr__(f'''_side_{m}''',
                             self._data.side_information.__getattribute__(f'''{self._loaders[m_id]}'''))

        all_multimodal_features = []
        for m_id, m in enumerate(self._modalities):
            all_multimodal_features.append(self.__getattribute__(
                f'''_side_{self._modalities[m_id]}''').object.get_all_features())

        self._model = BiVAECFModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            num_neg=self._num_neg,
            num_sample=self._num_sample,
            lr_lambda=self._lr_lambda,
            reg_weight=self._reg_weight,
            temperature=self._temperature,
            cap_item_priors=False, # TODO
            modalities=self._modalities,
            multimod_embed_k=self._multimod_factors,
            multimodal_features=all_multimodal_features,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "BiVAECF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        x = self._data.sp_i_train.copy()
        x.data = np.ones_like(x.data)  # Binarize data
        tx = x.transpose()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.num_items + self._data.num_users // self._batch_size), disable=not self._verbose) as t:
                for i_ids in self._sampler.item_iter(self._batch_size, shuffle=False):
                    steps += 1
                    loss += self._model.train_step(tx, i_ids, user=False)

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

                for u_ids in self._sampler.user_iter(self._batch_size, shuffle=False):
                    steps += 1
                    loss += self._model.train_step(x, u_ids, user=True)

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
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
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
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