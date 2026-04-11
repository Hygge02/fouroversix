#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("/root/fouroversix/results/lm_eval")
SUMMARY_PATH = RESULTS_ROOT / "aime24_sglang_qwen3_eval_summary.json"
POLL_SECONDS = 60


@dataclass
class EvalTarget:
    name: str
    output_root: Path


BASELINE = EvalTarget(
    name="baseline",
    output_root=RESULTS_ROOT / "aime24_sglang_baseline_qwen3_8b",
)
QUANT_SHARDS = [
    EvalTarget(
        name="quant_shard0",
        output_root=RESULTS_ROOT / "aime24_sglang_quant_shard0_qwen3_8b",
    ),
    EvalTarget(
        name="quant_shard1",
        output_root=RESULTS_ROOT / "aime24_sglang_quant_shard1_qwen3_8b",
    ),
    EvalTarget(
        name="quant_shard2",
        output_root=RESULTS_ROOT / "aime24_sglang_quant_shard2_qwen3_8b",
    ),
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def find_latest_results_file(output_root: Path) -> Path | None:
    candidates = sorted(output_root.glob("**/results_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def first_exact_match_metric(task_results: dict[str, Any]) -> tuple[str, float]:
    for key, value in task_results.items():
        if not isinstance(value, (int, float)):
            continue
        if "exact_match" not in key or "stderr" in key:
            continue
        return key, float(value)
    raise KeyError("No exact_match metric found in results")


def parse_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_results = payload["results"]["aime24"]
    metric_name, exact_match = first_exact_match_metric(task_results)
    sample_info = payload["n-samples"]["aime24"]
    return {
        "results_file": str(path),
        "metric_name": metric_name,
        "exact_match": exact_match,
        "sample_count": int(sample_info["effective"]),
        "raw_results": task_results,
    }


def collect_target(target: EvalTarget) -> dict[str, Any]:
    results_file = find_latest_results_file(target.output_root)
    if results_file is None:
        return {
            "status": "pending",
            "output_root": str(target.output_root),
        }

    parsed = parse_results(results_file)
    return {
        "status": "completed",
        "output_root": str(target.output_root),
        **parsed,
    }


def combine_quant(shards: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in shards if item.get("status") == "completed"]
    if len(completed) != len(shards):
        return {
            "status": "pending",
            "completed_shards": len(completed),
            "total_shards": len(shards),
        }

    total_samples = sum(item["sample_count"] for item in completed)
    weighted_exact_match = sum(
        item["exact_match"] * item["sample_count"] for item in completed
    ) / total_samples
    return {
        "status": "completed",
        "sample_count": total_samples,
        "exact_match": weighted_exact_match,
        "shards": [item["name"] for item in completed],
    }


def build_summary() -> dict[str, Any]:
    baseline = collect_target(BASELINE)
    quant_shards = []
    for shard in QUANT_SHARDS:
        shard_summary = collect_target(shard)
        shard_summary["name"] = shard.name
        quant_shards.append(shard_summary)

    quant_combined = combine_quant(quant_shards)

    comparison: dict[str, Any] = {"status": "pending"}
    if baseline.get("status") == "completed" and quant_combined.get("status") == "completed":
        comparison = {
            "status": "completed",
            "baseline_exact_match": baseline["exact_match"],
            "quant_exact_match": quant_combined["exact_match"],
            "delta_quant_minus_baseline": quant_combined["exact_match"] - baseline["exact_match"],
        }

    return {
        "updated_at": now_iso(),
        "task": "aime24",
        "models": {
            "baseline": "Qwen/Qwen3-8B",
            "quantized": "/root/fouroversix/artifacts/qwen3-8b-awq-fouroversix-weight-only",
        },
        "baseline": baseline,
        "quant_shards": quant_shards,
        "quant_combined": quant_combined,
        "comparison": comparison,
    }


def write_summary(summary: dict[str, Any]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def is_complete(summary: dict[str, Any]) -> bool:
    return summary["comparison"]["status"] == "completed"


def main() -> int:
    while True:
        summary = build_summary()
        write_summary(summary)
        print(
            json.dumps(
                {
                    "updated_at": summary["updated_at"],
                    "baseline_status": summary["baseline"]["status"],
                    "quant_status": summary["quant_combined"]["status"],
                    "summary_path": str(SUMMARY_PATH),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if is_complete(summary):
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
