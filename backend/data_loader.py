import os
import io
import requests
import numpy as np
import pandas as pd
from typing import Tuple, List

OPERATIONAL_SETTINGS = ["op1", "op2", "op3"]
SENSORS = [f"sensor{i}" for i in range(1, 22)]
COLUMN_NAMES = ["engine_id", "cycle"] + OPERATIONAL_SETTINGS + SENSORS


def load_raw_from_hf() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attempts to load FD003 dataset from Hugging Face datasets library."""
    from datasets import load_dataset
    ds = load_dataset("SoyVitou/NASA-C-MAPSS-Turbofan-Engine", "FD003")

    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])

    # Map HF column names to standard C-MAPSS names
    hf_mapping = {
        "unit_id": "engine_id",
        "setting_1": "op1",
        "setting_2": "op2",
        "setting_3": "op3",
    }
    # Map physical sensor names in HF to sensor1..sensor21
    hf_sensor_cols = [
        'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
        'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
        'PCNfR_dmd', 'W31', 'W32'
    ]
    for i, hf_col in enumerate(hf_sensor_cols):
        if i < 21:
            hf_mapping[hf_col] = f"sensor{i+1}"

    train_df.rename(columns=hf_mapping, inplace=True)
    test_df.rename(columns=hf_mapping, inplace=True)

    rul_df = None
    if "RUL" in test_df.columns:
        rul_df = test_df.groupby("engine_id")["RUL"].last().reset_index()

    # Drop non-standard columns if present (e.g. dataset, split)
    keep_cols = COLUMN_NAMES + (["RUL"] if "RUL" in train_df.columns else [])
    train_df = train_df[[c for c in keep_cols if c in train_df.columns]]
    test_df = test_df[[c for c in keep_cols if c in test_df.columns]]

    return train_df, test_df, rul_df


def load_raw_from_fallback() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fallback function to download raw C-MAPSS FD003 files from GitHub mirror if HuggingFace is unreachable."""
    cache_dir = os.path.join(os.path.dirname(__file__), "data_cache")
    os.makedirs(cache_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/tjbecker/CMAPSS-Dataset/master/"
    files = {
        "train": "train_FD003.txt",
        "test": "test_FD003.txt",
        "rul": "RUL_FD003.txt"
    }

    dfs = {}
    for key, filename in files.items():
        filepath = os.path.join(cache_dir, filename)
        if not os.path.exists(filepath):
            url = base_url + filename
            res = requests.get(url, timeout=30)
            if res.status_code == 200:
                with open(filepath, "w") as f:
                    f.write(res.text)
            else:
                raise RuntimeError(f"Failed to fetch {filename} from mirror {url}")

        if key == "rul":
            dfs[key] = pd.read_csv(filepath, sep=r"\s+", header=None, names=["RUL"])
        else:
            dfs[key] = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)

    return dfs["train"], dfs["test"], dfs["rul"]


def get_fd003_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], str]:
    """
    Main data loading entry point.
    Returns:
        train_df: Cleaned training DataFrame with RUL
        test_df: Cleaned test DataFrame with RUL
        rul_df: Ground truth test RUL DataFrame
        dropped_cols: List of dropped sensors/features with near-zero variance
        source_used: String description of dataset source used
    """
    source_used = "Hugging Face (SoyVitou/NASA-C-MAPSS-Turbofan-Engine)"
    try:
        train_df, test_df, rul_df = load_raw_from_hf()
    except Exception as e:
        print(f"Hugging Face load failed ({e}), using C-MAPSS FD003 mirror fallback...")
        train_df, test_df, rul_df = load_raw_from_fallback()
        source_used = "NASA C-MAPSS raw txt files (GitHub mirror fallback)"

    train_df["engine_id"] = train_df["engine_id"].astype(int)
    train_df["cycle"] = train_df["cycle"].astype(int)
    test_df["engine_id"] = test_df["engine_id"].astype(int)
    test_df["cycle"] = test_df["cycle"].astype(int)

    # Compute training set RUL if not present
    if "RUL" not in train_df.columns:
        max_cycles = train_df.groupby("engine_id")["cycle"].transform("max")
        train_df["RUL"] = max_cycles - train_df["cycle"]

    # Compute test set RUL if not present
    if "RUL" not in test_df.columns and rul_df is not None:
        ruls = rul_df["RUL"].values if "RUL" in rul_df.columns else rul_df.iloc[:, 0].values
        rul_dict = {i + 1: ruls[i] for i in range(len(ruls))}
        max_test_cycles = test_df.groupby("engine_id")["cycle"].transform("max")
        test_df["max_cycle"] = max_test_cycles
        test_df["final_rul"] = test_df["engine_id"].map(rul_dict)
        test_df["RUL"] = (test_df["max_cycle"] + test_df["final_rul"]) - test_df["cycle"]
        test_df.drop(columns=["max_cycle", "final_rul"], inplace=True)

    # Detect constant or near-zero variance features (variance < 1e-4)
    feature_cols = [c for c in OPERATIONAL_SETTINGS + SENSORS if c in train_df.columns]
    std_devs = train_df[feature_cols].std()
    dropped_cols = std_devs[std_devs < 1e-4].index.tolist()

    # Drop zero/near-zero variance columns
    train_df.drop(columns=dropped_cols, errors="ignore", inplace=True)
    test_df.drop(columns=dropped_cols, errors="ignore", inplace=True)

    return train_df, test_df, rul_df, dropped_cols, source_used


if __name__ == "__main__":
    tr, te, ru, dropped, src = get_fd003_data()
    print(f"Loaded dataset from: {src}")
    print(f"Train shape: {tr.shape}, Test shape: {te.shape}")
    print(f"Dropped sensors/features: {dropped}")
