from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# print(f"root:{PROJECT_ROOT} src:{SRC_PATH}")

from polystock.dimensionality import compare_domains_pca


def parse_drop_cols(value: str | None) -> list[str] | None:
    # print(f"raw_value={value}")
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    # print("Building CLI parser")
    parser = argparse.ArgumentParser(
        description="Compare PCA dimensionality between stock and polymarket datasets"
    )
    parser.add_argument("--stock-csv", required=True, help="Path to stock CSV file")
    parser.add_argument(
        "--polymarket-csv", required=True, help="Path to polymarket CSV file"
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.95,
        help="Cumulative explained variance threshold (default: 0.95)",
    )
    parser.add_argument(
        "--scaler",
        choices=["standard", "robust"],
        default="standard",
        help="Feature scaling strategy",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Optional max number of PCA components to fit",
    )
    parser.add_argument(
        "--stock-drop-cols",
        type=str,
        default=None,
        help="Comma-separated non-numeric/id/date columns to drop for stock data",
    )
    parser.add_argument(
        "--polymarket-drop-cols",
        type=str,
        default=None,
        help="Comma-separated non-numeric/id/date columns to drop for polymarket data",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path",
    )

    args = parser.parse_args()
    # print(f"Parsed args={args}")

    result = compare_domains_pca(
        stock_csv_path=args.stock_csv,
        polymarket_csv_path=args.polymarket_csv,
        variance_threshold=args.variance_threshold,
        scaler=args.scaler,
        max_components=args.max_components,
        stock_drop_cols=parse_drop_cols(args.stock_drop_cols),
        polymarket_drop_cols=parse_drop_cols(args.polymarket_drop_cols),
    )
    # print("PCA comparison finished")

    payload = json.dumps(result, indent=2)
    # print(f"Payload length={len(payload)} chars")

    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        # print(f"Wrote output JSON to {output_path}")
        print(f"Saved PCA comparison to {output_path}")
    else:
        # print("Printing payload to stdout")
        print(payload)


if __name__ == "__main__":
    main()
