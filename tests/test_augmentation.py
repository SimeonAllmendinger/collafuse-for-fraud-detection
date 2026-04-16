from __future__ import annotations

import unittest
import warnings

import pandas as pd

from src.collafuse.augmentation import (
    compute_target_synthetic_count,
    generate_adasyn_samples,
    generate_smote_samples,
)


class AugmentationTest(unittest.TestCase):
    def test_ratio_math_and_oversamplers_support_amplification(self) -> None:
        frame = pd.DataFrame(
            {
                "f1": [0.10, 0.18, 0.24, 0.28, 0.32, 0.36, 0.42, 0.48],
                "f2": [1.0, 1.2, 1.4, 1.8, 2.0, 2.2, 2.4, 2.6],
                "isFraud": [1, 1, 1, 0, 0, 0, 0, 0],
            }
        )
        target_count = compute_target_synthetic_count(frame["isFraud"], ratio=1.5)
        self.assertEqual(target_count, 3)
        smote_samples = generate_smote_samples(frame, "isFraud", target_count=target_count, random_state=7, n_neighbors=2)
        adasyn_samples = generate_adasyn_samples(frame, "isFraud", target_count=target_count, random_state=7, n_neighbors=2)
        self.assertEqual(len(smote_samples), target_count)
        self.assertEqual(len(adasyn_samples), target_count)

    def test_oversamplers_do_not_raise_fragmentation_warning(self) -> None:
        frame = pd.DataFrame(
            {
                **{f"f{i}": [float(i + row) for row in range(8)] for i in range(64)},
                "isFraud": [1, 1, 1, 0, 0, 0, 0, 0],
            }
        )
        target_count = compute_target_synthetic_count(frame["isFraud"], ratio=1.5)

        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            smote_samples = generate_smote_samples(
                frame,
                "isFraud",
                target_count=target_count,
                random_state=7,
                n_neighbors=2,
            )
            adasyn_samples = generate_adasyn_samples(
                frame,
                "isFraud",
                target_count=target_count,
                random_state=7,
                n_neighbors=2,
            )

        self.assertEqual(len(smote_samples), target_count)
        self.assertEqual(len(adasyn_samples), target_count)


if __name__ == "__main__":
    unittest.main()
