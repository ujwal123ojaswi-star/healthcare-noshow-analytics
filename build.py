"""Build everything end to end:  python build.py

Loads (or generates) appointment data, engineers features, trains + evaluates
the models, and saves the model plus scored appointments. Then launch the
dashboard with:  streamlit run app/dashboard.py
"""
from __future__ import annotations

from features import build_features
from ingest.load import load_appointments
from model import train


def main() -> None:
    print("==> [1/3] Loading appointment data")
    df = load_appointments()

    print("\n==> [2/3] Engineering features")
    feats = build_features(df)
    print(f"[build] {len(feats):,} rows, no-show rate {feats['no_show'].mean():.1%}")

    print("\n==> [3/3] Training + evaluating models")
    train(feats)

    print("\nDone. Launch the dashboard with:")
    print("    streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
