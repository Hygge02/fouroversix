from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--output-dir",
        default="/root/fouroversix/results/qwen3_8b_layerwise_3d_raw",
    )
    parser.add_argument(
        "--max-points-per-axis",
        type=int,
        default=192,
        help="Maximum sampled points to render along each axis.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use a locally cached Hugging Face snapshot.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return (
        name.replace("/", "__")
        .replace(".", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def sampled_indices(size: int, max_points: int) -> tuple[np.ndarray, int]:
    stride = max(1, math.ceil(size / max_points))
    indices = np.arange(0, size, stride, dtype=np.int64)
    if indices[-1] != size - 1:
        indices = np.append(indices, size - 1)
    return indices, stride


def plot_tensor(
    name: str,
    tensor,
    output_path: Path,
    max_points_per_axis: int,
) -> dict[str, object]:
    rows, cols = tensor.shape
    row_idx, row_stride = sampled_indices(rows, max_points_per_axis)
    col_idx, col_stride = sampled_indices(cols, max_points_per_axis)

    sampled = tensor[row_idx][:, col_idx].float().cpu().numpy()
    x_grid, y_grid = np.meshgrid(row_idx, col_idx, indexing="ij")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        sampled,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
        shade=True,
    )
    ax.set_xlabel("row")
    ax.set_ylabel("column")
    ax.set_zlabel("value")
    ax.set_title(
        f"{name}\nshape={rows}x{cols}, sampled={sampled.shape[0]}x{sampled.shape[1]}, "
        f"row_stride={row_stride}, col_stride={col_stride}"
    )
    fig.colorbar(surface, shrink=0.6, pad=0.08, label="value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "name": name,
        "shape": [rows, cols],
        "sampled_shape": [int(sampled.shape[0]), int(sampled.shape[1])],
        "row_stride": row_stride,
        "col_stride": col_stride,
        "min": float(sampled.min()),
        "max": float(sampled.max()),
        "mean": float(sampled.mean()),
        "std": float(sampled.std()),
        "path": str(output_path),
    }


def render_index(output_dir: Path, plots: list[dict[str, object]]) -> None:
    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>Qwen3-8B Layerwise 3D Weight Plots</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 24px; }",
        ".item { margin-bottom: 28px; }",
        "img { max-width: 960px; width: 100%; border: 1px solid #ddd; }",
        "code { background: #f5f5f5; padding: 2px 4px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Qwen3-8B Layerwise 3D Weight Plots</h1>",
        "<p>Signed raw values on sampled row/column coordinates. No absolute value transform.</p>",
    ]

    for item in plots:
        rel_path = Path(item["path"]).name
        lines.extend(
            [
                '<div class="item">',
                f"<h3>{html.escape(str(item['name']))}</h3>",
                (
                    f"<p>shape=<code>{item['shape'][0]}x{item['shape'][1]}</code>, "
                    f"sampled=<code>{item['sampled_shape'][0]}x{item['sampled_shape'][1]}</code>, "
                    f"row_stride=<code>{item['row_stride']}</code>, "
                    f"col_stride=<code>{item['col_stride']}</code>, "
                    f"min=<code>{item['min']:.6g}</code>, "
                    f"max=<code>{item['max']:.6g}</code></p>"
                ),
                f'<a href="{html.escape(rel_path)}"><img src="{html.escape(rel_path)}" alt="{html.escape(str(item["name"]))}"></a>',
                "</div>",
            ]
        )

    lines.extend(["</body>", "</html>"])
    (output_dir / "index.html").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            args.model_id,
            local_files_only=args.local_files_only,
        )
    )

    plots: list[dict[str, object]] = []

    weight_files = sorted(snapshot_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found under {snapshot_path}")

    plot_idx = 0
    for file_path in weight_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for tensor_name in f.keys():
                tensor = f.get_tensor(tensor_name)
                if not tensor.is_floating_point() or tensor.ndim != 2:
                    continue

                output_path = output_dir / f"{plot_idx:03d}_{sanitize_name(tensor_name)}.png"
                info = plot_tensor(
                    tensor_name,
                    tensor,
                    output_path,
                    args.max_points_per_axis,
                )
                info["source_file"] = file_path.name
                plots.append(info)
                plot_idx += 1
                print(
                    f"[{plot_idx:03d}] rendered {tensor_name} "
                    f"shape={tuple(tensor.shape)} -> {output_path.name}",
                    flush=True,
                )

    metadata = {
        "model_id": args.model_id,
        "snapshot_path": str(snapshot_path),
        "max_points_per_axis": args.max_points_per_axis,
        "num_plots": len(plots),
        "plots": plots,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    render_index(output_dir, plots)
    print(f"done: {len(plots)} plots -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
