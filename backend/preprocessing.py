import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

SEQUENCE_LENGTH = 30
EARLY_CYCLES_THRESHOLD = 30  # Healthy phase window cutoff


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Returns operational and sensor column names present in DataFrame."""
    exclude = {"engine_id", "cycle", "RUL"}
    return [col for col in df.columns if col not in exclude]


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> MinMaxScaler:
    """Fits a MinMaxScaler on the training features."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def transform_dataframe(df: pd.DataFrame, feature_cols: List[str], scaler: MinMaxScaler) -> pd.DataFrame:
    """Transforms DataFrame features using fitted scaler."""
    scaled_df = df.copy()
    scaled_df[feature_cols] = scaler.transform(df[feature_cols])
    return scaled_df


def extract_healthy_data(scaled_train_df: pd.DataFrame, max_cycle: int = EARLY_CYCLES_THRESHOLD) -> pd.DataFrame:
    """Filters training dataset to healthy initial cycles."""
    return scaled_train_df[scaled_train_df["cycle"] <= max_cycle].copy()


def create_padded_sequences_for_engine(
    engine_df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sequence windows (N_cycles, seq_length, n_features) for a single engine.
    Pads earlier cycles (< seq_length) by prepending the cycle 1 feature vector.
    """
    features = engine_df[feature_cols].values
    cycles = engine_df["cycle"].values
    n_samples = len(features)

    sequences = []
    for i in range(n_samples):
        start_idx = max(0, i + 1 - seq_length)
        sub_seq = features[start_idx : i + 1]
        pad_len = seq_length - len(sub_seq)
        if pad_len > 0:
            pad_block = np.repeat(sub_seq[0:1], pad_len, axis=0)
            padded_seq = np.vstack([pad_block, sub_seq])
        else:
            padded_seq = sub_seq
        sequences.append(padded_seq)

    return np.array(sequences), cycles


def prepare_windowed_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates sequence dataset for all engines in DataFrame.
    Returns:
        X: np.ndarray (total_windows, seq_length, n_features)
        engine_ids: np.ndarray (total_windows,)
        cycles: np.ndarray (total_windows,)
    """
    X_list = []
    engine_ids_list = []
    cycles_list = []

    for eng_id, group in df.groupby("engine_id"):
        seqs, cycs = create_padded_sequences_for_engine(group, feature_cols, seq_length)
        X_list.append(seqs)
        engine_ids_list.append(np.full(len(cycs), eng_id))
        cycles_list.append(cycs)

    X = np.vstack(X_list)
    engine_ids = np.concatenate(engine_ids_list)
    cycles = np.concatenate(cycles_list)

    return X, engine_ids, cycles
