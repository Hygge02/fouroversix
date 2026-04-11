#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from lm_eval.tasks.aime.utils import process_results


STOP_STRINGS = ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"]


@dataclass(frozen=True)
class ServerSpec:
    port: int
    model: str
    attention_backend: str | None
    decode_attention_backend: str | None
    prefill_attention_backend: str | None
    page_size: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", required=True, help="Comma-separated SGLang ports.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--output-dir",
        default="/root/fouroversix/results/sglang_full_eval_qwen3_8b_unquantized_aime24",
    )
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--request-timeout", type=int, default=999999)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_server_spec(port: int) -> ServerSpec:
    info = request_json(f"http://127.0.0.1:{port}/server_info")
    return ServerSpec(
        port=port,
        model=str(info.get("model_path")),
        attention_backend=info.get("attention_backend"),
        decode_attention_backend=info.get("decode_attention_backend"),
        prefill_attention_backend=info.get("prefill_attention_backend"),
        page_size=info.get("page_size"),
    )


def build_prompt(problem: str) -> str:
    return f"Question: {problem}\nAnswer:"


def evaluate_one(
    server: ServerSpec,
    doc: dict[str, Any],
    model: str,
    request_timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": build_prompt(doc["Problem"]),
        "max_tokens": 32768,
        "temperature": 0.0,
        "stop": STOP_STRINGS,
    }
    url = f"http://127.0.0.1:{server.port}/v1/completions"

    last_error = None
    for attempt in range(1, max_retries + 1):
        start = time.perf_counter()
        try:
            response = request_json(url, payload=payload, timeout=request_timeout)
            elapsed = time.perf_counter() - start
            choice = response["choices"][0]
            text = choice.get("text", "")
            usage = response.get("usage", {})
            exact_match = int(process_results(doc, [text])["exact_match"])
            completion_tokens = int(usage.get("completion_tokens", 0))
            return {
                "status": "ok",
                "attempt": attempt,
                "elapsed_seconds": elapsed,
                "completion_tokens": completion_tokens,
                "throughput_tokens_per_second": (
                    completion_tokens / elapsed if elapsed > 0 else None
                ),
                "finish_reason": choice.get("finish_reason"),
                "matched_stop": choice.get("matched_stop"),
                "exact_match": exact_match,
                "response_text": text,
                "server_port": server.port,
                "server_attention_backend": server.attention_backend,
                "server_decode_attention_backend": server.decode_attention_backend,
                "server_prefill_attention_backend": server.prefill_attention_backend,
                "server_page_size": server.page_size,
                "usage": usage,
            }
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            last_error = {
                "attempt": attempt,
                "elapsed_seconds": elapsed,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            time.sleep(min(attempt, 5))

    return {
        "status": "error",
        "server_port": server.port,
        "server_attention_backend": server.attention_backend,
        "server_decode_attention_backend": server.decode_attention_backend,
        "server_prefill_attention_backend": server.prefill_attention_backend,
        "server_page_size": server.page_size,
        "elapsed_seconds": (
            float(last_error["elapsed_seconds"]) if last_error is not None else 0.0
        ),
        "exact_match": 0,
        "response_text": None,
        "completion_tokens": 0,
        "throughput_tokens_per_second": None,
        "finish_reason": None,
        "matched_stop": None,
        "usage": {},
        "last_error": last_error,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    ports = [int(item) for item in args.ports.split(",") if item.strip()]
    servers = [get_server_spec(port) for port in ports]

    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    docs = [dict(dataset[idx]) for idx in range(len(dataset))]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.json"
    summary_path = output_dir / "summary.json"

    start_wall = time.perf_counter()
    started_at = utc_now()

    work_queue: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
    for idx, doc in enumerate(docs):
        work_queue.put((idx, doc))

    results: list[dict[str, Any] | None] = [None] * len(docs)
    result_lock = threading.Lock()
    file_lock = threading.Lock()

    def flush_partial() -> None:
        with result_lock:
            completed = [item for item in results if item is not None]
            exact_match_sum = sum(int(item["exact_match"]) for item in completed)
            total_tokens = sum(int(item["completion_tokens"]) for item in completed)
            total_request_seconds = sum(float(item["elapsed_seconds"]) for item in completed)
            payload = {
                "updated_at": utc_now(),
                "started_at": started_at,
                "model": args.model,
                "dataset": "Maxwell-Jia/AIME_2024",
                "split": "train",
                "num_samples_total": len(results),
                "num_samples_completed": len(completed),
                "exact_match_completed_only": (
                    exact_match_sum / len(completed) if completed else None
                ),
                "total_completion_tokens_completed": total_tokens,
                "total_request_seconds_completed": total_request_seconds,
                "servers": [server.__dict__ for server in servers],
                "results": completed,
            }
        with file_lock:
            write_json(details_path, payload)

    def worker(server: ServerSpec) -> None:
        while True:
            try:
                idx, doc = work_queue.get_nowait()
            except queue.Empty:
                return

            item_result = evaluate_one(
                server=server,
                doc=doc,
                model=args.model,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            item_result.update(
                {
                    "doc_index": idx,
                    "doc_id": doc["ID"],
                    "answer": str(doc["Answer"]),
                    "problem": doc["Problem"],
                }
            )
            with result_lock:
                results[idx] = item_result
            print(
                json.dumps(
                    {
                        "event": "sample_done",
                        "doc_index": idx,
                        "doc_id": doc["ID"],
                        "answer": str(doc["Answer"]),
                        "exact_match": item_result["exact_match"],
                        "elapsed_seconds": item_result["elapsed_seconds"],
                        "completion_tokens": item_result["completion_tokens"],
                        "finish_reason": item_result["finish_reason"],
                        "server_port": server.port,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            flush_partial()
            work_queue.task_done()

    threads = [threading.Thread(target=worker, args=(server,), daemon=True) for server in servers]
    flush_partial()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ended_at = utc_now()
    total_wall_seconds = time.perf_counter() - start_wall
    completed = [item for item in results if item is not None]
    exact_match_sum = sum(int(item["exact_match"]) for item in completed)
    total_completion_tokens = sum(int(item["completion_tokens"]) for item in completed)
    total_request_seconds = sum(float(item["elapsed_seconds"]) for item in completed)

    summary = {
        "started_at": started_at,
        "ended_at": ended_at,
        "model": args.model,
        "dataset": "Maxwell-Jia/AIME_2024",
        "split": "train",
        "num_samples": len(completed),
        "correct": exact_match_sum,
        "exact_match": exact_match_sum / len(completed),
        "total_wall_seconds": total_wall_seconds,
        "total_completion_tokens": total_completion_tokens,
        "overall_wall_throughput_tokens_per_second": (
            total_completion_tokens / total_wall_seconds if total_wall_seconds > 0 else None
        ),
        "aggregate_request_seconds": total_request_seconds,
        "servers": [server.__dict__ for server in servers],
        "details_path": str(details_path),
    }
    with file_lock:
        write_json(summary_path, summary)
    flush_partial()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
