from __future__ import annotations

import logging
from pathlib import Path

from src.collafuse.preprocessing import (
    apply_global_template,
    assign_clients,
    fit_global_template,
    save_template_artifacts,
    split_client_frame,
)
from src.collafuse.visualization import plot_preparation_client_overview
from src.config_files.configs import PaperPipelineConfig
from src.pipeline.datasets import load_preparation_dataset
from src.pipeline.common import ensure_directory, read_json, write_json

LOGGER = logging.getLogger("collafuse")


def _client_is_viable(train_frame, test_frame, label_column: str) -> bool:
    return (
        not train_frame.empty
        and not test_frame.empty
        and train_frame[label_column].nunique() > 1
        and test_frame[label_column].nunique() > 1
    )


def _existing_prepared_data_is_usable(prepared_root: Path) -> bool:
    metadata_path = prepared_root / "metadata.json"
    if not metadata_path.exists():
        return False

    metadata = read_json(metadata_path)
    client_entries = metadata.get("clients", [])
    if not client_entries:
        return False

    for client_entry in client_entries:
        train_path = Path(client_entry["train_path"])
        test_path = Path(client_entry["test_path"])
        if not train_path.exists() or not test_path.exists():
            return False

    return True


def run_preparation(
    config: PaperPipelineConfig,
    raw_transaction_path: str | Path | None = None,
    raw_identity_path: str | Path | None = None,
    raw_main_path: str | Path | None = None,
    raw_aux_path: str | Path | None = None,
    raw_edge_path: str | Path | None = None,
) -> Path:
    prepared_root = ensure_directory(config.paths.prepared_root)
    if config.data.use_prepared_if_exists and _existing_prepared_data_is_usable(prepared_root):
        LOGGER.info("Reusing existing prepared data at %s", prepared_root)
        return prepared_root

    clients_dir = ensure_directory(prepared_root / "clients")

    loaded_dataset = load_preparation_dataset(
        config,
        raw_transaction_path=raw_transaction_path,
        raw_identity_path=raw_identity_path,
        raw_main_path=raw_main_path,
        raw_aux_path=raw_aux_path,
        raw_edge_path=raw_edge_path,
    )
    assigned = assign_clients(loaded_dataset.frame, config.client_split)
    client_index = (
        assigned[[config.client_split.client_column, config.client_split.client_label_column]]
        .drop_duplicates()
        .sort_values(config.client_split.client_column)
        .reset_index(drop=True)
    )
    if client_index.empty:
        raise ValueError(f"No clients were created for dataset {config.data.dataset_name}")

    reference_client_id = config.data.encoder_reference_client_id
    if reference_client_id not in client_index[config.client_split.client_column].tolist():
        reference_client_id = str(client_index.iloc[0][config.client_split.client_column])
        LOGGER.warning(
            "Encoder reference client %s is unavailable for %s; using %s instead",
            config.data.encoder_reference_client_id,
            config.data.dataset_name,
            reference_client_id,
        )

    reference_frame = assigned.loc[
        assigned[config.client_split.client_column] == reference_client_id
    ].reset_index(drop=True)
    template, encoder, scaler = fit_global_template(reference_frame, config.data, config.client_split)
    template_paths = save_template_artifacts(prepared_root, template, encoder, scaler)

    client_entries: list[dict[str, str | int]] = []
    skipped_clients: list[dict[str, str | int]] = []
    for client_row in client_index.to_dict(orient="records"):
        client_id = str(client_row[config.client_split.client_column])
        client_label = str(client_row[config.client_split.client_label_column])
        client_frame = assigned.loc[assigned[config.client_split.client_column] == client_id].reset_index(drop=True)
        if client_frame.empty:
            continue
        train_frame, test_frame = split_client_frame(client_frame, config.data)
        if not _client_is_viable(train_frame, test_frame, config.data.label_column):
            skipped_clients.append(
                {
                    "client_id": client_id,
                    "client_label": client_label,
                    "rows": int(len(client_frame)),
                    "fraud_rows": int((client_frame[config.data.label_column] == 1).sum()),
                }
            )
            LOGGER.warning(
                "Skipping client %s during preparation because the split is not viable for training/evaluation",
                client_id,
            )
            continue

        processed_train = apply_global_template(
            train_frame,
            template,
            encoder,
            scaler,
            config.data,
            config.client_split
        )
        processed_test = apply_global_template(test_frame, template, encoder, scaler, config.data, config.client_split)

        train_path = clients_dir / f"{client_id}_train.csv"
        test_path = clients_dir / f"{client_id}_test.csv"
        processed_train.to_csv(train_path, index=False)
        processed_test.to_csv(test_path, index=False)

        client_entries.append(
            {
                "client_id": client_id,
                "client_label": client_label,
                "train_path": str(train_path),
                "test_path": str(test_path),
                "train_rows": int(len(processed_train)),
                "test_rows": int(len(processed_test)),
                "train_fraud": int((processed_train[config.data.label_column] == 1).sum()),
                "test_fraud": int((processed_test[config.data.label_column] == 1).sum()),
            }
        )

    metadata = {
        "dataset_name": config.data.dataset_name,
        "prepared_root": str(prepared_root),
        "label_column": config.data.label_column,
        "feature_columns": template.final_columns,
        "clients": client_entries,
        "skipped_clients": skipped_clients,
        **loaded_dataset.source_paths,
        **template_paths,
    }
    if not client_entries:
        raise ValueError(
            f"No viable clients remained after preparation for dataset {config.data.dataset_name}. "
            "Adjust the client split strategy or provide a larger dataset."
        )
    client_overview_paths = plot_preparation_client_overview(
        dataset_name=config.data.dataset_name,
        client_entries=client_entries,
        skipped_clients=skipped_clients,
        output_dir=prepared_root,
    )
    metadata["client_rows_overview_path"] = client_overview_paths["rows"]
    metadata["client_fraud_overview_path"] = client_overview_paths["fraud"]
    write_json(prepared_root / "metadata.json", metadata)
    assigned[
        [config.client_split.client_column, config.client_split.client_label_column]
        ].value_counts().reset_index(name="count").to_csv(
        prepared_root / "client_distribution.csv",
        index=False,
    )
    return prepared_root
