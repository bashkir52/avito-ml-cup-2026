import unittest

import numpy as np

from intent_features import intent_features
from ranking import INTENT


class IntentFeatureTests(unittest.TestCase):
    def test_nearest_event_keeps_its_original_position_type_and_source(self):
        table = np.array([[5, 0], [0, 5]], dtype=np.int8)
        row = {'item_id__embedding_idx': [0, 1, -1], 'category_id': [3, 8, 3],
               'history__item_id__embedding_idx': [-1, 1, 0],
               'history__event_type': [9, 2, 4], 'history__event_source': [1, 3, 7],
               'history__category_id': [8, 8, 3]}
        features = intent_features(row, table)
        col = {name: i for i, name in enumerate(INTENT)}
        for name, expected in [('intent_match_event_rank', [2, 1]), ('intent_match_type', [4, 2]),
                               ('intent_match_source', [7, 3]), ('intent_max_type_4', [1, 0])]:
            np.testing.assert_equal(features[:2, col[name]], expected)
        self.assertTrue(np.isnan(features[2, col['intent_match_type']]))
        self.assertTrue(np.isnan(features[:, col['intent_max_type_9']]).all())

    def test_empty_history_and_missing_candidate_vectors(self):
        row = {'item_id__embedding_idx': [-1], 'category_id': [3],
               'history__item_id__embedding_idx': []}
        features = intent_features(row, np.array([[5, 0]], dtype=np.int8))
        self.assertTrue(np.isnan(features).all())


if __name__ == '__main__':
    unittest.main()
