import unittest

import numpy as np
import pyarrow as pa

from history_retrieval import history_examples, candidate_masks


class HistoryRetrievalTest(unittest.TestCase):
    def test_chronological_targets_and_duplicate_mask(self):
        ids = [[91, 92, 91, 93, 93, 5, 6], [10, 11, 12, 13, 14, 15]]
        batch = pa.record_batch([ids, ids, [[1]*len(row) for row in ids],
                                 [[2]*len(row) for row in ids], [21, 22]],
                                names=['history__item_id', 'history__item_id__embedding_idx',
                                       'history__event_type', 'history__category_id', 'user_id'])
        example = history_examples(batch, np.ones(2, dtype=bool))
        np.testing.assert_array_equal(example['target_id'], [92, 91, 10, 11, 12, 13])
        np.testing.assert_array_equal(example['context_index'][0, :4], [93, 5, 6, -1])
        np.testing.assert_array_equal(example['target_owner'], [0, 0, 1, 1, 1, 1])

    def test_other_known_history_items_and_same_user_are_not_negatives(self):
        users = np.array([99, 22, 99])
        histories = np.array([[0, 7, 10], [0, 8, 12], [0, 6, 9]])
        item = np.array([7, 8, 10, 9])
        owner = np.array([0, 1, 1, 2])
        own, valid = candidate_masks(histories, users, item, owner)
        np.testing.assert_array_equal(own[0], [True, False, False, False])
        np.testing.assert_array_equal(valid[0], [True, True, False, False])
        np.testing.assert_array_equal(valid[1], [True, True, True, True])


if __name__ == '__main__':
    unittest.main()
