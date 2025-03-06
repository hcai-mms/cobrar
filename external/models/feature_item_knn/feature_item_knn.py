"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import time

from elliot.recommender.recommender_utils_mixin import RecMixin

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from .feature_item_knn_similarity import Similarity

from ast import literal_eval as make_tuple
import wandb


class FeatureItemKNN(RecMixin, BaseRecommenderModel):
    r"""
    Attribute Item-kNN proposed in MyMediaLite Recommender System Library

    For further details, please refer to the `paper <https://www.researchgate.net/publication/221141162_MyMediaLite_A_free_recommender_system_library>`_

    Args:
        neighbors: Number of item neighbors
        similarity: Similarity function

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        FeatureItemKNN:
          meta:
            save_recs: True
          neighbors: 40
          similarity: cosine
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_modal_sim_factor", "modal_sim_factor", "msf", 0.5, float, None),
            ("_implicit", "implicit", "bin", False, None, None),
            ("_modalities", "modalities", "modalites", "('visual','textual')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_aggregation", "aggregation", "aggregation", 'mean', str, None),
            ("_loaders", "loaders", "loads", "('VisualAttribute','TextualAttribute')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-"))
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        for m_id, m in enumerate(self._modalities):
            self.__setattr__(f'''_side_{m}''',
                             self._data.side_information.__getattribute__(f'''{self._loaders[m_id]}'''))

        all_multimodal_features = []
        for m_id, m in enumerate(self._modalities):
            all_multimodal_features.append(self.__getattribute__(
                f'''_side_{self._modalities[m_id]}''').object.get_all_features())

        self._model = Similarity(data=self._data,
                                 num_neighbors=self._num_neighbors,
                                 similarity=self._similarity,
                                 implicit=self._implicit,
                                 modalities=self._modalities,
                                 multimodal_features=all_multimodal_features,
                                 aggregation=self._aggregation,
                                 modal_sim_factor=self._modal_sim_factor)

        wandb.init(
            project=f"FeatureItemKNN-{config.data_config.dataset_path.split('/')[-2]}",
            name=self.name,
            config={
                **{
                    "neighbors": self._num_neighbors,
                    "similarity": self._similarity,
                    "implicit": self._implicit,
                    "modalities": self._modalities,
                    "aggregation": self._aggregation,
                    "modal_sim_factor": self._modal_sim_factor
                }
            },
            reinit=True,
        )

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._ratings.keys()}

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    @property
    def name(self):
        return f"FeatureItemKNN_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        print(f"Transactions: {self._data.transactions}")

        self.evaluate()

