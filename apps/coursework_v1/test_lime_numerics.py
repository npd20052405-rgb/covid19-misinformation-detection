"""Exercise the LIME call boundary under batch-dependent numerical variation."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xai_lime as x


class LimeNumerics(unittest.TestCase):
    def run_case(self, original, batched):
        original=np.array(original,dtype=float)
        saved=dict(probability=original[None,:],labels=np.array([['fake','real'][original.argmax()]]),truncated=[False])
        def neighborhood(model_id,texts,deadline=None):
            return np.tile(batched,(len(texts),1))
        with patch.object(x.engine,'predict',return_value=saved), patch.object(x,'lime_probabilities',side_effect=neighborhood):
            return x.explain('word_svm_cal','The hospital reported coronavirus cases.',num_samples=200)

    def test_small_batch_roundoff_is_anchored(self):
        result=self.run_case([.983304,.016696],[.983288,.016712])
        self.assertAlmostEqual(result['prob_real'],.016696)
        self.assertAlmostEqual(result['batch_anchor_max_difference'],.000016)

    def test_material_pipeline_difference_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'pipeline'):
            self.run_case([.8,.2],[.7,.3])

    def test_even_small_label_flip_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'nhãn'):
            self.run_case([.50001,.49999],[.49999,.50001])


if __name__=='__main__':unittest.main(verbosity=2)
