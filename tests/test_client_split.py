from __future__ import annotations

import unittest

import pandas as pd

from src.config_files.configs import ClientSplitConfig
from src.collafuse.preprocessing import assign_clients


class ClientSplitTest(unittest.TestCase):
    def test_client_mapping_and_fallback(self) -> None:
        frame = pd.DataFrame(
            {
                "card4": ["mastercard", "visa", "mastercard", "visa", "discover"],
                "card6": ["credit", "debit", "debit", "credit", "credit"],
            }
        )
        mapped = assign_clients(frame, ClientSplitConfig())
        self.assertEqual(mapped["client_id"].tolist(), ["CLIENT_1", "CLIENT_2", "CLIENT_3", "CLIENT_4", "CLIENT_0"])

    def test_categorical_client_mapping_uses_explicit_values_and_fallback(self) -> None:
        frame = pd.DataFrame({"type": ["PAYMENT", "TRANSFER", "DEBIT", "UNKNOWN"]})
        mapped = assign_clients(
            frame,
            ClientSplitConfig(
                strategy="categorical",
                source_column="type",
                categorical_values=["PAYMENT", "TRANSFER", "DEBIT"],
            ),
        )
        self.assertEqual(mapped["client_id"].tolist(), ["CLIENT_0", "CLIENT_1", "CLIENT_2", "CLIENT_3"])
        self.assertEqual(mapped["client_label"].tolist(), ["type_payment", "type_transfer", "type_debit", "type_other"])

    def test_quantile_client_mapping_creates_multiple_bins(self) -> None:
        frame = pd.DataFrame({"Time": [0, 10, 20, 30, 40, 50, 60, 70]})
        mapped = assign_clients(
            frame,
            ClientSplitConfig(
                strategy="quantile",
                source_column="Time",
                num_clients=4,
            ),
        )
        self.assertEqual(sorted(mapped["client_id"].unique().tolist()), ["CLIENT_0", "CLIENT_1", "CLIENT_2", "CLIENT_3"])


if __name__ == "__main__":
    unittest.main()
