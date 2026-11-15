"""Customer segmentation with KMeans.

Steps: load CSV -> scale -> try k=2..8 -> pick best k -> label -> plots -> report.

Usage:
    python src/segment.py                      # uses data/customers.csv
    python src/segment.py path/to/your.csv     # use your own data
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, plots are saved to files
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
K_VALUES = range(2, 9)
RANDOM_STATE = 42


def load_data(path):
    """Read the CSV and return the dataframe plus the numeric feature columns."""
    df = pd.read_csv(path)
    features = [c for c in df.select_dtypes("number").columns if "id" not in c.lower()]
    if not features:
        raise ValueError("No numeric feature columns found in the CSV.")
    df = df.dropna(subset=features).reset_index(drop=True)
    return df, features


def scale_features(df, features):
    return StandardScaler().fit_transform(df[features])


def find_best_k(X, k_values=K_VALUES):
    """Run KMeans for each k and return inertias, silhouette scores and the best k."""
    inertias, silhouettes = [], []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, model.labels_))
    best_k = list(k_values)[silhouettes.index(max(silhouettes))]
    return inertias, silhouettes, best_k


def plot_k_selection(inertias, silhouettes, out_path, k_values=K_VALUES):
    ks = list(k_values)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ks, inertias, "o-")
    axes[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
    axes[1].plot(ks, silhouettes, "o-", color="green")
    axes[1].set(title="Silhouette score", xlabel="k", ylabel="Score")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def name_segment(row, overall):
    """Give a simple readable name based on income and spending vs the average."""
    income = "High income" if row["AnnualIncome_k"] >= overall["AnnualIncome_k"] else "Low income"
    spend = "high spending" if row["SpendingScore"] >= overall["SpendingScore"] else "low spending"
    return f"{income}, {spend}"


def write_report(profile, df, features, best_k, best_score, out_path):
    overall = df[features].mean()
    lines = ["# Segment Insights", "",
             f"Clusters (k): {best_k}", f"Silhouette score: {best_score:.3f}", ""]
    for cluster, row in profile.iterrows():
        size = int(row["Size"])
        lines.append(f"## Cluster {cluster}: {name_segment(row, overall)}")
        lines.append(f"- Customers: {size} ({size / len(df):.0%})")
        for f in features:
            diff = (row[f] - overall[f]) / overall[f] * 100
            lines.append(f"- {f}: {row[f]:.1f} ({diff:+.0f}% vs overall average)")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Customer segmentation using KMeans")
    parser.add_argument("csv", nargs="?", default=str(ROOT / "data" / "customers.csv"))
    parser.add_argument("--out", default=str(ROOT / "outputs"), help="output folder")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. load
    df, features = load_data(args.csv)
    print(f"[1] Loaded {len(df)} rows, features: {features}")

    # 2. scale
    X = scale_features(df, features)
    print("[2] Features scaled with StandardScaler")

    # 3. choose k and fit final model
    inertias, silhouettes, best_k = find_best_k(X)
    best_score = max(silhouettes)
    print(f"[3] Best k = {best_k} (silhouette {best_score:.3f})")
    plot_k_selection(inertias, silhouettes, out / "k_selection.png")
    model = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X)

    # 4. labels
    df["Cluster"] = model.labels_
    df.to_csv(out / "customers_labeled.csv", index=False)
    print("[4] Cluster sizes:")
    print(df["Cluster"].value_counts().sort_index().to_string())

    # 5. pair plot + report
    grid = sns.pairplot(df[features + ["Cluster"]], hue="Cluster", palette="tab10", corner=True)
    grid.fig.suptitle("Customer clusters - pair plot", y=1.02)
    grid.savefig(out / "pairplot.png", dpi=130)
    plt.close("all")

    profile = df.groupby("Cluster")[features].mean().round(1)
    profile["Size"] = df["Cluster"].value_counts().sort_index()
    profile.to_csv(out / "cluster_profile.csv")
    write_report(profile, df, features, best_k, best_score, out / "insights.md")
    print("[5] Cluster profile:")
    print(profile.to_string())
    print(f"Done. Results saved in {out}")


if __name__ == "__main__":
    main()
