import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_data import make_customers
from segment import find_best_k, scale_features


def test_data_shape_and_columns():
    df = make_customers()
    assert df.shape == (250, 5)
    assert "CustomerID" in df.columns


def test_scaling_gives_zero_mean():
    df = make_customers()
    feats = ["Age", "AnnualIncome_k", "SpendingScore", "AnnualVisits"]
    X = scale_features(df, feats)
    assert abs(X.mean()) < 1e-6


def test_best_k_in_range():
    df = make_customers()
    feats = ["Age", "AnnualIncome_k", "SpendingScore", "AnnualVisits"]
    _, scores, best_k = find_best_k(scale_features(df, feats))
    assert 2 <= best_k <= 8
    assert max(scores) > 0.3
