from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from item_priors import DAY, START, accumulate, prior_features


class ItemPriorsTests(unittest.TestCase):
    def test_only_complete_prior_days_contribute(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            impressions = root / 'train' / 'impressions'
            impressions.mkdir(parents=True)
            pq.write_table(pa.table({'timestamp': [START + 30, START + DAY + 30],
                                     'item_id': [[1, 2], [2, 1]],
                                     'target': [[0, 1], [0, 1]]}), impressions / 'part-0.parquet')
            mapping = np.array([-1, 0, 1, 2])
            with patch('item_priors.DATA', root):
                exposure, positive = accumulate(mapping, 3)

        data = {'timestamp': np.array([START + 40, START + DAY + 40, START + 2 * DAY]),
                'counts': np.array([2, 2, 3]), 'item': np.array([1, 2, 1, 2, 1, 2, 3])}
        result = prior_features(data, mapping, exposure, positive)
        np.testing.assert_allclose(result[:, 0], np.log1p([0, 0, 1, 1, 2, 2, 0]))
        np.testing.assert_allclose(result[:, 1], np.log1p([0, 0, 0, 1, 1, 1, 0]))
        np.testing.assert_allclose(result[[0, 1, 6], 2], .02)


if __name__ == '__main__':
    unittest.main()
