from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any


def load_checkpoint_metadata(path: Path) -> tuple[bool, str, str]:
    if not path.exists():
        return False, "", "missing"
    if importlib.util.find_spec("torch") is None:
        return False, "", "torch_not_available"
    import torch  # noqa: PLC0415

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return False, "", f"load_failed: {type(exc).__name__}: {exc}"
    if not isinstance(checkpoint, dict):
        return False, "", "not_a_dict"
    if "model_state_dict" not in checkpoint:
        return False, str(checkpoint.get("epoch", "")), "missing_model_state_dict"
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict) or not state:
        return False, str(checkpoint.get("epoch", "")), "empty_model_state_dict"
    return True, str(checkpoint.get("epoch", "")), "ok"


def infer_task(run_name: str) -> str:
    name = run_name.lower()
    if "burgers" in name:
        return "Burgers"
    if "v1e3" in name or "1e-3" in name:
        return "Navier-Stokes v1e-3"
    if "v1e4" in name or "1e-4" in name:
        return "Navier-Stokes v1e-4"
    if "shallow" in name:
        return "Shallow Water"
    return "unknown"


def row_for_run(run_dir: Path) -> dict[str, Any]:
    checkpoint = run_dir / "checkpoint_last.pt"
    ok, epoch, status = load_checkpoint_metadata(checkpoint)
    args_path = run_dir / "args.json"
    config_path = run_dir / "config.yaml"
    return {
        "task": infer_task(run_dir.name),
        "run_name": run_dir.name,
        "checkpoint_path": checkpoint if checkpoint.exists() else "",
        "checkpoint_epoch": epoch,
        "loadable": "yes" if ok else "no",
        "status": status,
        "args_path": args_path if args_path.exists() else "",
        "config_path": config_path if config_path.exists() else "",
    }


def write_markdown(path: Path, rows: list[dict[str, Any]], outputs_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 0 KNO Checkpoint Inventory",
        "",
        "本文件只登记真正可用于 evaluation-only 的 checkpoint。`time_error.pt`、误差张量和 `metrics.csv` 不计入 checkpoint。",
        "",
        f"扫描目录：`{outputs_root}`",
        "",
        "| task | run name | checkpoint path | checkpoint epoch | loadable | status | args/config |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        args_config = f"{row['args_path']}<br>{row['config_path']}"
        lines.append(
            "| {task} | `{run_name}` | `{checkpoint_path}` | {checkpoint_epoch} | {loadable} | {status} | {args_config} |".format(
                **row,
                args_config=args_config,
            )
        )
    if not rows:
        lines.append("| none | none | none |  | no | no_run_dirs | none |")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- `loadable=yes` 才能进入独立评估脚本。",
            "- `missing_model_state_dict` 表示文件即使存在也不能算作模型 checkpoint。",
            "- 在 checkpoint 状态确认前，不应启动长时间重训。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Inventory Stage 0 KNO checkpoints")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/stage0_kno_baseline"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/stage0_kno_baseline/checkpoint_inventory.md"),
    )
    args = parser.parse_args()

    rows = [row_for_run(path) for path in sorted(args.outputs_root.iterdir()) if path.is_dir()]
    write_markdown(args.out, rows, args.outputs_root)
    print(f"wrote {args.out}")
    for row in rows:
        print(f"{row['run_name']}: {row['loadable']} ({row['status']})")


if __name__ == "__main__":
    main()
