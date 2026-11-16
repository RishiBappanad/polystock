from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from polystock.dimensionality import compare_domains_pca


def parse_drop_cols(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PCA dimensionality comparison on processed NASDAQ + Polymarket feature files"
    )
    parser.add_argument(
        "--stock-csv",
        default=str(PROJECT_ROOT / "data/processed/nasdaq_cleaned_features.csv"),
        help="Path to processed stock CSV",
    )
    parser.add_argument(
        "--polymarket-csv",
        default=str(PROJECT_ROOT / "data/processed/polymarket_cleaned_features.csv"),
        help="Path to processed polymarket CSV",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.95,
        help="Cumulative explained variance threshold",
    )
    parser.add_argument(
        "--scaler",
        choices=["standard", "robust"],
        default="standard",
        help="Scaling strategy",
    )
    parser.add_argument(
        "--stock-drop-cols",
        default="date,ticker,Target_Log_Return",
        help="Comma-separated columns to drop from stock data",
    )
    parser.add_argument(
        "--polymarket-drop-cols",
        default="Date,market_id,conditionId,umaResolutionStatus,Target_Deviation,final_settlement",
        help="Comma-separated columns to drop from polymarket data",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "reports/pca_processed_compare.json"),
        help="Output JSON path",
    )

    args = parser.parse_args()

    result = compare_domains_pca(
        stock_csv_path=args.stock_csv,
        polymarket_csv_path=args.polymarket_csv,
        variance_threshold=args.variance_threshold,
        scaler=args.scaler,
        stock_drop_cols=parse_drop_cols(args.stock_drop_cols),
        polymarket_drop_cols=parse_drop_cols(args.polymarket_drop_cols),
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Saved PCA comparison to {output_path}")
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
