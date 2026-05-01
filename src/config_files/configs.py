from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DatasetName = Literal["ieee_cis", "baf", "paysim", "credit_card_fraud", "elliptic"]
BaselineSource = Literal[
    "real_only_unweighted",
    "real_only_weighted",
    "random_oversampling",
    "smote",
    "adasyn",
    "collafuse",
    "ctgan",
    "tabddpm",
    "local_only_ddpm",
    "centralized_ddpm",
]


def default_client_rules() -> list["ClientRule"]:
    return [
        ClientRule(client_id="CLIENT_0", label="other", fallback=True),
        ClientRule(client_id="CLIENT_1", label="mastercard_credit", card4="mastercard", card6="credit"),
        ClientRule(client_id="CLIENT_2", label="visa_debit", card4="visa", card6="debit"),
        ClientRule(client_id="CLIENT_3", label="mastercard_debit", card4="mastercard", card6="debit"),
        ClientRule(client_id="CLIENT_4", label="visa_credit", card4="visa", card6="credit"),
    ]


def default_classifier_specs() -> list["ClassifierSpec"]:
    return [
        ClassifierSpec(name="logistic_regression"),
        ClassifierSpec(name="fedavg_logistic_regression"),
        ClassifierSpec(name="random_forest"),
        ClassifierSpec(name="hist_gradient_boosting"),
        ClassifierSpec(name="lightgbm"),
    ]


def default_baseline_sources() -> list[BaselineSource]:
    return [
        "real_only_unweighted",
        "real_only_weighted",
        "random_oversampling",
        "smote",
        "adasyn",
        "collafuse",
        "ctgan",
        "tabddpm",
        "local_only_ddpm",
        "centralized_ddpm",
    ]


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_transaction_path: Path = Path("data/raw/train_transaction.csv")
    raw_identity_path: Path = Path("data/raw/train_identity.csv")
    raw_main_path: Path | None = None
    raw_aux_path: Path | None = None
    raw_edge_path: Path | None = None
    prepared_root: Path = Path("artifacts/prepared")
    stage1_root: Path = Path("artifacts/stage1")
    stage2_root: Path = Path("artifacts/stage2")
    run_all_stages_root: Path = Path("artifacts/run_all_stages")


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: DatasetName = "ieee_cis"
    transaction_id_column: str = "TransactionID"
    label_column: str = "isFraud"
    use_prepared_if_exists: bool = False
    fill_strategy: Literal["mean", "median", "zero"] = "median"
    test_size: float = 0.15
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    col_nan_threshold: float = 0.5
    row_nan_threshold: float = 0.5
    encoder_reference_client_id: str = "CLIENT_0"
    shuffle_seed: int = 38
    drop_columns: list[str] = Field(default_factory=list)

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        return value

    @field_validator("row_nan_threshold")
    @classmethod
    def validate_row_nan_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("row_nan_threshold must be between 0 and 1")
        return value


class ClientRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str
    label: str
    card4: str | None = None
    card6: str | None = None
    fallback: bool = False

    @model_validator(mode="after")
    def validate_rule(self) -> "ClientRule":
        if not self.fallback and (self.card4 is None or self.card6 is None):
            raise ValueError("non-fallback client rules require card4 and card6")
        return self


class ClientSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["card_rules", "categorical", "quantile"] = "card_rules"
    source_column: str | None = None
    num_clients: int = 5
    categorical_values: list[str] = Field(default_factory=list)
    card4_column: str = "card4"
    card6_column: str = "card6"
    client_column: str = "client_id"
    client_label_column: str = "client_label"
    rules: list[ClientRule] = Field(default_factory=default_client_rules)

    @field_validator("num_clients")
    @classmethod
    def validate_num_clients(cls, value: int) -> int:
        if value < 1:
            raise ValueError("num_clients must be at least 1")
        return value

    @model_validator(mode="after")
    def validate_strategy(self) -> "ClientSplitConfig":
        if self.strategy == "card_rules":
            self.fallback_rule
        elif not self.source_column:
            raise ValueError("source_column is required for categorical and quantile client split strategies")

        if self.strategy == "categorical" and len(self.categorical_values) != len(set(self.categorical_values)):
            raise ValueError("categorical_values must be unique")
        return self

    @property
    def fallback_rule(self) -> ClientRule:
        for rule in self.rules:
            if rule.fallback:
                return rule
        raise ValueError("at least one fallback client rule is required")


class CollaFuseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    num_timesteps: int = 1000
    tau_ratio: float = 0.2
    batch_size: int = 1000
    epochs: int = 300
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    time_embed_dim: int = 128
    checkpoint_every: int = 25
    w1: float = 0.1
    w2: float = 0.1
    num_workers: int = 0

    @field_validator("tau_ratio")
    @classmethod
    def validate_tau_ratio(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("tau_ratio must be between 0 and 1")
        return value


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples_per_client: int = 5000
    sample_batch_size: int = 500
    random_seed: int = 38
    n_inference_timesteps: int | None = None
    target_label: int = 1


class AugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ratios: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0])
    smote_k_neighbors: int = 5
    adasyn_n_neighbors: int = 5

    @field_validator("ratios")
    @classmethod
    def validate_ratios(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("at least one augmentation ratio is required")
        if any(ratio < 0 for ratio in value):
            raise ValueError("augmentation ratios must be non-negative")
        return value


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classifier_seeds: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 38])
    mmd_seeds: list[int] = Field(default_factory=lambda: list(range(30)))
    pca_components: int = 10
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float | Literal["auto"] = "auto"
    tsne_random_state: int = 38
    mmd_batch_size: int = 1000


class BaselinesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled_sources: list[BaselineSource] = Field(default_factory=default_baseline_sources)
    ddpm_epochs: int | None = None
    ddpm_batch_size: int | None = None
    ctgan_epochs: int = 50
    ctgan_batch_size: int = 500
    ctgan_generator_dim: list[int] = Field(default_factory=lambda: [256, 256])
    ctgan_discriminator_dim: list[int] = Field(default_factory=lambda: [256, 256])

    @field_validator("enabled_sources")
    @classmethod
    def validate_enabled_sources(cls, value: list[BaselineSource]) -> list[BaselineSource]:
        if not value:
            raise ValueError("at least one baseline source is required")
        return value


class ClassifierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["logistic_regression", "fedavg_logistic_regression", "random_forest", "hist_gradient_boosting", "lightgbm"]
    params: dict[str, Any] = Field(default_factory=dict)


class ClassifiersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: list[ClassifierSpec] = Field(default_factory=default_classifier_specs)


class PaperPipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: PathConfig = Field(default_factory=PathConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    client_split: ClientSplitConfig = Field(default_factory=ClientSplitConfig)
    collafuse: CollaFuseConfig = Field(default_factory=CollaFuseConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)
    classifiers: ClassifiersConfig = Field(default_factory=ClassifiersConfig)
