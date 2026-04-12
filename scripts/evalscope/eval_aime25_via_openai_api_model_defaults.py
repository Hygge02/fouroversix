from __future__ import annotations

import argparse

from evalscope.config import TaskConfig
from evalscope.run import run_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_cfg = TaskConfig(
        model=args.model,
        model_id=args.model_id,
        datasets=["aime25"],
        api_url=args.api_url,
        api_key=args.api_key,
        eval_type="openai_api",
        work_dir=args.work_dir,
        no_timestamp=True,
        enable_progress_tracker=True,
        limit=args.limit,
        # Keep the request free of generation overrides so the server-side
        # model defaults remain authoritative.
        generation_config={
            "temperature": None,
            "max_tokens": None,
            "top_p": None,
            "top_k": None,
            "extra_body": None,
        },
    )
    run_task(task_cfg)


if __name__ == "__main__":
    main()
