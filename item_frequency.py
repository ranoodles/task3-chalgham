from __future__ import annotations

import argparse
from pathlib import Path

import hdbscan
import pandas as pd
from sentence_transformers import SentenceTransformer


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_clustered{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster phrases in a CSV column using HDBSCAN and append the "
            "cluster frequency for each row."
        )
    )
    parser.add_argument(
        "--input",
        default="/Users/anshsinghal/Desktop/task3-git/task3-chalgham/ai_degradation_evidence_new.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output CSV file. Defaults to *_clustered.csv.",
    )
    parser.add_argument(
        "--columns",
        default="AI Features,Performance Degradation Types,Causal Links",
        help=(
            "Comma-separated list of CSV columns to cluster. "
            "Defaults to: AI Features, Performance Degradation Types, "
            "Causal Links."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default="Cluster Frequency",
        help="Suffix to append to each column name for frequency output.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="HDBSCAN min_cluster_size parameter.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples parameter (default: None).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _default_output_path(input_path)

    df = pd.read_csv(input_path)
    columns = [col.strip() for col in args.columns.split(",") if col.strip()]
    columns = [col for col in columns if col != "AI Feature Measurements"]
    if not columns:
        raise ValueError("No valid columns provided.")

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in {input_path}: {', '.join(missing)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    for column in columns:
        phrases = df[column].fillna("").astype(str).tolist()
        embeddings = model.encode(
            phrases, show_progress_bar=True, normalize_embeddings=True
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)

        cluster_sizes = pd.Series(labels).value_counts().to_dict()
        output_column = f"{column} {args.output_suffix}"
        df[output_column] = [cluster_sizes[label] for label in labels]

    df.to_csv(output_path, index=False)
    print(f"Wrote clustered CSV to {output_path}")


if __name__ == "__main__":
    main()

