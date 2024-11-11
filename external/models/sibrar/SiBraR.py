import torch
import os
import numpy as np
from tqdm import tqdm
from ast import literal_eval as make_tuple

from .custom_sampler import Sampler
from elliot.utils.write import store_recommendation

from elliot.recommender import BaseRecommenderModel
from .SiBraRModel import SiBraRModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger
import math

# sp_i_train_ratings = self._data.sp_i_train_ratings,
# random_seed = self._seed,
# item_multimodal_features = all_multimodal_features,  # actual tensors

class SiBraR(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # Define Parameters as tuples: (variable_name, public_name, shortcut, default, reading_function, printing_function)
        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_input_dim", "input_dim", "input_dim", 256, int, None),
            ("_factors", "factors", "factors", 64, int, None),
            # ("_reg_weight", "reg_weight", "reg_weight", 0.01, float, None),
            # ("_combine_modalities", "comb_mod", "comb_mod", 'none', str, None),
            ("_cl_weight", "cl_weight", "cl_weight", 0.01, float, None),
            ("_use_user_profile", "use_user_profile", "use_user_profile", True, bool, None),
            ("_cl_temperature", "cl_temperature", "cl_temperature", 0.01, float, None),
            ("_item_modalities", "item_modalities", "item_modalites", "('visual','textual')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_loaders", "loaders", "loads", "('VisualAttribute','TextualAttribute')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-"))
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        # def __init__(self, indexed_ratings, transactions, batch_size, all_items, seed=42):

        self._sampler = Sampler(self._data.i_train_dict,
                                self._data.transactions,
                                self._batch_size,
                                self._data.edge_index['itemId'].unique().tolist(),
                                self._seed)

        for m_id, m in enumerate(self._item_modalities):
            self.__setattr__(f'''_side_{m}''',
                             self._data.side_information.__getattribute__(f'''{self._loaders[m_id]}'''))

        all_multimodal_features = {m_id: self.__getattribute__(
                    f'''_side_{self._item_modalities[m_id]}''').object.get_all_features() for m_id, m in enumerate(self._item_modalities)}

        self._model = SiBraRModel(
                 # item multimodal features should also include interactions
                 num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            input_dim=self._input_dim,
            cl_weight=self._cl_weight,
            cl_temperature=self._cl_temperature,
            embed_k=self._factors,
            sp_i_train_ratings=self._data.sp_i_train_ratings,
            item_modalities=self._item_modalities,
            use_user_profile=self._use_user_profile,
            item_multimodal_features=all_multimodal_features,  # dictionary of actual tensors
            random_seed=self._seed,
        )


    @property
    def name(self):
        return "SiBraR" \
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
                for batch in self._sampler.step():
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

        item_repr = torch.mean(item_repr, dim=-2)
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