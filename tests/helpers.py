from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.config_files.configs import (
    BaselinesConfig,
    ClassifierSpec,
    ClassifiersConfig,
    CollaFuseConfig,
    EvaluationConfig,
    PaperPipelineConfig,
    PathConfig,
    SamplingConfig,
    AugmentationConfig,
    DataConfig,
    ClientSplitConfig,
)

TestDatasetName = Literal["ieee_cis", "baf", "paysim", "credit_card_fraud", "elliptic"]


def create_toy_ieee_dataset(base_dir: Path) -> tuple[Path, Path]:
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    transaction_path = raw_dir / "train_transaction.csv"
    identity_path = raw_dir / "train_identity.csv"

    groups = [
        ("american express", "credit"),
        ("mastercard", "credit"),
        ("visa", "debit"),
        ("mastercard", "debit"),
        ("visa", "credit"),
    ]
    transaction_rows = []
    identity_rows = []
    transaction_id = 1
    for group_index, (card4, card6) in enumerate(groups):
        for row_index in range(12):
            is_fraud = 1 if row_index in {1, 5, 9} else 0
            transaction_rows.append(
                {
                    "TransactionID": transaction_id,
                    "isFraud": is_fraud,
                    "card4": card4,
                    "card6": card6,
                    "TransactionAmt": float((group_index + 1) * 100 + row_index * 3 + is_fraud * 20),
                    "dist1": float(group_index * 10 + row_index),
                    "V1": float((row_index % 4) * 0.5 + group_index),
                    "ProductCD": ["W", "C", "H"][row_index % 3],
                }
            )
            identity_rows.append(
                {
                    "TransactionID": transaction_id,
                    "DeviceType": "mobile" if row_index % 2 else "desktop",
                    "id_01": float(group_index + row_index / 10.0),
                }
            )
            transaction_id += 1

    pd.DataFrame(transaction_rows).to_csv(transaction_path, index=False)
    pd.DataFrame(identity_rows).to_csv(identity_path, index=False)
    return transaction_path, identity_path


def create_toy_paysim_dataset(base_dir: Path) -> Path:
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = raw_dir / "paysim.csv"
    types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    rows = []
    for group_index, tx_type in enumerate(types):
        for row_index in range(12):
            is_fraud = 1 if row_index in {1, 6, 10} else 0
            amount = float(50 + group_index * 25 + row_index * 3 + is_fraud * 40)
            rows.append(
                {
                    "step": group_index * 24 + row_index,
                    "type": tx_type,
                    "amount": amount,
                    "nameOrig": f"C{group_index:02d}{row_index:04d}",
                    "oldbalanceOrg": float(1000 + group_index * 100 + row_index * 10),
                    "newbalanceOrig": float(900 + group_index * 100 + row_index * 8 - is_fraud * 35),
                    "nameDest": f"M{group_index:02d}{row_index:04d}",
                    "oldbalanceDest": float(500 + group_index * 120 + row_index * 5),
                    "newbalanceDest": float(520 + group_index * 120 + row_index * 6 + is_fraud * 45),
                    "isFraud": is_fraud,
                    "isFlaggedFraud": int(is_fraud and row_index % 2 == 0),
                }
            )
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    return dataset_path


def create_toy_credit_card_dataset(base_dir: Path) -> Path:
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = raw_dir / "creditcard.csv"
    rows = []
    for row_index in range(60):
        is_fraud = 1 if row_index % 11 in {1, 7} else 0
        rows.append(
            {
                "Time": row_index * 100,
                "V1": round(-2.5 + row_index * 0.08 + is_fraud * 0.6, 4),
                "V2": round(1.0 + (row_index % 5) * 0.2 - is_fraud * 0.3, 4),
                "V3": round(0.5 + (row_index % 7) * 0.15 + is_fraud * 0.4, 4),
                "Amount": float(10 + (row_index % 9) * 12 + is_fraud * 80),
                "Class": is_fraud,
            }
        )
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    return dataset_path


def create_toy_baf_dataset(base_dir: Path) -> Path:
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = raw_dir / "baf.csv"
    rows = []
    for row_index in range(60):
        month = row_index % 12
        is_fraud = 1 if row_index % 10 in {2, 8} else 0
        rows.append(
            {
                "month": month,
                "income": float(1500 + row_index * 35),
                "payment_type": ["CARD", "TRANSFER", "DIRECT_DEBIT"][row_index % 3],
                "employment_status": ["employed", "self-employed", "student"][row_index % 3],
                "velocity_6h": float((row_index % 6) + is_fraud * 2),
                "fraud_bool": is_fraud,
            }
        )
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    return dataset_path


def create_toy_elliptic_dataset(base_dir: Path) -> tuple[Path, Path, Path]:
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    features_path = raw_dir / "elliptic_txs_features.csv"
    classes_path = raw_dir / "elliptic_txs_classes.csv"
    edge_path = raw_dir / "elliptic_txs_edgelist.csv"

    feature_rows = []
    class_rows = []
    edge_rows = []
    for row_index in range(60):
        tx_id = 1000 + row_index
        time_step = (row_index // 6) + 1
        class_label = "1" if row_index % 10 in {1, 8} else "2"
        feature_rows.append(
            [
                tx_id,
                time_step,
                round(0.1 * row_index, 4),
                round(1.5 + (row_index % 5) * 0.3, 4),
                round(2.0 + (row_index % 7) * 0.25 + (class_label == "1") * 0.5, 4),
                round(3.0 + (row_index % 11) * 0.2, 4),
            ]
        )
        class_rows.append({"txId": tx_id, "class": class_label})
        if row_index:
            edge_rows.append({"txId1": tx_id - 1, "txId2": tx_id})

    pd.DataFrame(feature_rows).to_csv(features_path, index=False, header=False)
    pd.DataFrame(class_rows).to_csv(classes_path, index=False)
    pd.DataFrame(edge_rows).to_csv(edge_path, index=False)
    return features_path, classes_path, edge_path


def build_test_config(base_dir: Path, dataset_name: TestDatasetName = "ieee_cis") -> PaperPipelineConfig:
    path_config = PathConfig(
        prepared_root=base_dir / "artifacts" / "prepared",
        stage1_root=base_dir / "artifacts" / "stage1",
        stage2_root=base_dir / "artifacts" / "stage2",
        run_all_stages_root=base_dir / "artifacts" / "run_all_stages",
    )
    data_config = DataConfig(dataset_name=dataset_name, test_size=0.25, shuffle_seed=7, variance_threshold=0.0)
    client_split = ClientSplitConfig()

    if dataset_name == "ieee_cis":
        transaction_path, identity_path = create_toy_ieee_dataset(base_dir)
        path_config = path_config.model_copy(update={"raw_transaction_path": transaction_path, "raw_identity_path": identity_path})
        data_config = data_config.model_copy(update={"transaction_id_column": "TransactionID", "label_column": "isFraud"})
        client_split = ClientSplitConfig()
    elif dataset_name == "paysim":
        dataset_path = create_toy_paysim_dataset(base_dir)
        path_config = path_config.model_copy(update={"raw_main_path": dataset_path})
        data_config = data_config.model_copy(
            update={
                "transaction_id_column": "paysim_transaction_id",
                "label_column": "isFraud",
                "drop_columns": ["nameOrig", "nameDest"],
            }
        )
        client_split = ClientSplitConfig(
            strategy="categorical",
            source_column="type",
            categorical_values=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        )
    elif dataset_name == "credit_card_fraud":
        dataset_path = create_toy_credit_card_dataset(base_dir)
        path_config = path_config.model_copy(update={"raw_main_path": dataset_path})
        data_config = data_config.model_copy(update={"transaction_id_column": "credit_card_transaction_id", "label_column": "Class"})
        client_split = ClientSplitConfig(strategy="quantile", source_column="Time", num_clients=5)
    elif dataset_name == "baf":
        dataset_path = create_toy_baf_dataset(base_dir)
        path_config = path_config.model_copy(update={"raw_main_path": dataset_path})
        data_config = data_config.model_copy(update={"transaction_id_column": "baf_record_id", "label_column": "fraud_bool"})
        client_split = ClientSplitConfig(strategy="quantile", source_column="month", num_clients=5)
    elif dataset_name == "elliptic":
        features_path, classes_path, edge_path = create_toy_elliptic_dataset(base_dir)
        path_config = path_config.model_copy(
            update={"raw_main_path": features_path, "raw_aux_path": classes_path, "raw_edge_path": edge_path}
        )
        data_config = data_config.model_copy(update={"transaction_id_column": "txId", "label_column": "class"})
        client_split = ClientSplitConfig(strategy="quantile", source_column="time_step", num_clients=5)
    else:
        raise ValueError(f"Unsupported test dataset: {dataset_name}")

    return PaperPipelineConfig(
        paths=path_config,
        data=data_config,
        client_split=client_split,
        collafuse=CollaFuseConfig(
            device="cpu",
            num_timesteps=12,
            tau_ratio=0.2,
            batch_size=4,
            epochs=1,
            learning_rate=1e-3,
            lr_warmup_steps=1,
            time_embed_dim=16,
            checkpoint_every=1,
            num_workers=0,
        ),
        sampling=SamplingConfig(samples_per_client=8, sample_batch_size=4, random_seed=7, n_inference_timesteps=12),
        augmentation=AugmentationConfig(ratios=[0.2, 1.5], smote_k_neighbors=2, adasyn_n_neighbors=2),
        evaluation=EvaluationConfig(classifier_seeds=[1], mmd_seeds=[0, 1], pca_components=3, tsne_perplexity=5.0, mmd_batch_size=4),
        baselines=BaselinesConfig(
            ddpm_epochs=1,
            ddpm_batch_size=4,
            ctgan_epochs=1,
            ctgan_batch_size=10,
            ctgan_generator_dim=[32, 32],
            ctgan_discriminator_dim=[32, 32],
        ),
        classifiers=ClassifiersConfig(
            suite=[
                ClassifierSpec(name="logistic_regression", params={"max_iter": 200}),
                ClassifierSpec(
                    name="fedavg_logistic_regression",
                    params={"num_rounds": 2, "local_epochs": 1, "learning_rate": 0.05, "batch_size": 4},
                ),
                ClassifierSpec(name="random_forest", params={"n_estimators": 10, "min_samples_leaf": 1}),
                ClassifierSpec(name="hist_gradient_boosting", params={"max_iter": 20, "max_depth": 3}),
            ]
        ),
    )
