from pathlib import Path
import pytest

from preprocessing.loader import load_pems04


def test_pems04_loader():
    dataset_dir = Path("data/raw/PEMS04")
    npz_file = dataset_dir / "PEMS04.npz"
    csv_file = dataset_dir / "PEMS04.csv"

    if not npz_file.exists() or not csv_file.exists():
        pytest.skip("PEMS04 dataset not available in CI environment")

    data, edges, info = load_pems04("data/raw")

    assert data.shape == (16992, 307, 3)
    assert edges.shape[0] == 340
    assert info["time_steps"] == 16992
    assert info["nodes"] == 307
    assert info["features"] == 3
