from __future__ import annotations

import csv
from pathlib import Path


def _float_or_inf(value: str | None) -> float:
    if value in (None, ""):
        return float("inf")
    try:
        return float(value)
    except ValueError:
        return float("inf")


def main() -> None:
    rows: list[dict[str, str]] = []
    for metric_path in sorted(Path("outputs").glob("*/metrics.csv")):
        with metric_path.open(newline="") as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        last = data[-1]
        best = min(
            data,
            key=lambda r: _float_or_inf(
                r.get("full_rel_l2")
                or r.get("test_full_rel_l2")
                or r.get("rel_l2")
                or r.get("test_rel_l2")
            ),
        )
        rows.append(
            {
                "run_name": metric_path.parent.name,
                "last_epoch": last.get("epoch", ""),
                "best_epoch": best.get("epoch", ""),
                "best_full_rel_l2": best.get(
                    "full_rel_l2", best.get("test_full_rel_l2", "")
                ),
                "best_step_rel_l2": best.get(
                    "step_rel_l2", best.get("test_step_rel_l2", "")
                ),
                "notes": "",
            }
        )

    out = Path("results/run_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "last_epoch",
                "best_epoch",
                "best_full_rel_l2",
                "best_step_rel_l2",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
