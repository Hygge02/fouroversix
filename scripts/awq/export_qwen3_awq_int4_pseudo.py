from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from awq.quantize.pre_quant import apply_awq, run_awq
from awq.quantize.quantizer import pseudo_quantize_model_weight
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--awq-cache-path", default=None)
    parser.add_argument("--awq-group-size", type=int, default=128)
    parser.add_argument("--awq-n-samples", type=int, default=128)
    parser.add_argument("--awq-seqlen", type=int, default=512)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16"],
    )
    parser.add_argument("--calib-data", default="wikitext")
    parser.add_argument(
        "--zero-point",
        action="store_true",
        help="Enable asymmetric quantization with learned zero points.",
    )
    parser.add_argument(
        "--preserve-max-abs",
        action="store_true",
        help=(
            "Only for symmetric quantization: keep the maximum-absolute-weight value "
            "in each group unquantized and quantize the remaining values."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def default_awq_cache_path(
    model_name: str,
    *,
    awq_group_size: int,
    awq_n_samples: int,
    awq_seqlen: int,
    zero_point: bool,
    preserve_max_abs: bool,
    dtype_name: str,
) -> Path:
    model_slug = model_name.replace("/", "__")
    return Path("ptq/awq") / model_slug / (
        f"w4_g{awq_group_size}_zp{int(zero_point)}_pm{int(preserve_max_abs)}"
        f"_n{awq_n_samples}_s{awq_seqlen}_{dtype_name}.pt"
    )


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    msg = f"Unsupported dtype: {name}"
    raise ValueError(msg)


def load_model_and_tokenizer(
    model_name: str,
    *,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer


def load_or_run_awq(
    model_name: str,
    cache_path: Path,
    *,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    awq_group_size: int,
    awq_n_samples: int,
    awq_seqlen: int,
    calib_data: str,
    zero_point: bool,
    preserve_max_abs: bool,
) -> dict:
    if cache_path.exists():
        print(f"Loading cached AWQ results from {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu")

    print("Loading source model for AWQ calibration...", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    start = time.perf_counter()
    awq_results = run_awq(
        model,
        tokenizer,
        w_bit=16,
        q_config={
            "q_group_size": awq_group_size,
            "zero_point": zero_point,
            "preserve_max_abs": preserve_max_abs,
        },
        n_samples=awq_n_samples,
        seqlen=awq_seqlen,
        calib_data=calib_data,
    )
    elapsed = time.perf_counter() - start
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(awq_results, cache_path)
    print(
        f"Saved AWQ calibration results to {cache_path} in {elapsed:.2f} seconds",
        flush=True,
    )
    return awq_results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.force:
            msg = f"{output_dir} already exists; pass --force to overwrite."
            raise FileExistsError(msg)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = resolve_torch_dtype(args.dtype)
    asymmetric = args.zero_point
    if args.preserve_max_abs and args.zero_point:
        msg = "--preserve-max-abs only supports symmetric quantization."
        raise ValueError(msg)
    awq_cache_path = (
        Path(args.awq_cache_path)
        if args.awq_cache_path is not None
        else default_awq_cache_path(
            args.model_name,
            awq_group_size=args.awq_group_size,
            awq_n_samples=args.awq_n_samples,
            awq_seqlen=args.awq_seqlen,
            zero_point=args.zero_point,
            preserve_max_abs=args.preserve_max_abs,
            dtype_name=args.dtype,
        )
    )

    total_start = time.perf_counter()
    awq_results = load_or_run_awq(
        args.model_name,
        awq_cache_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        awq_group_size=args.awq_group_size,
        awq_n_samples=args.awq_n_samples,
        awq_seqlen=args.awq_seqlen,
        calib_data=args.calib_data,
        zero_point=args.zero_point,
        preserve_max_abs=args.preserve_max_abs,
    )

    print("Reloading source model for pseudo quantization...", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    print("Applying AWQ scales/clips...", flush=True)
    apply_awq(model, awq_results)

    print(
        "Running pseudo quantization with "
        f"{'asymmetric' if asymmetric else 'symmetric'} INT4 "
        f"(group_size={args.awq_group_size}, zero_point={args.zero_point}, "
        f"preserve_max_abs={args.preserve_max_abs})...",
        flush=True,
    )
    pseudo_quantize_model_weight(
        model,
        w_bit=args.w_bit,
        q_config={
            "q_group_size": args.awq_group_size,
            "zero_point": args.zero_point,
            "preserve_max_abs": args.preserve_max_abs,
        },
    )

    print(f"Saving pseudo-quantized model to {output_dir}", flush=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["awq_pseudo_quantization_config"] = {
        "quant_method": "awq_pseudo_quantized",
        "bits": args.w_bit,
        "group_size": args.awq_group_size,
        "zero_point": args.zero_point,
        "asymmetric": asymmetric,
        "preserve_max_abs": args.preserve_max_abs,
        "n_samples": args.awq_n_samples,
        "seqlen": args.awq_seqlen,
        "calib_data": args.calib_data,
        "dtype": args.dtype,
        "awq_cache_path": str(awq_cache_path.resolve()),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    summary = {
        "model_name": args.model_name,
        "output_dir": str(output_dir),
        "awq_cache_path": str(awq_cache_path.resolve()),
        "w_bit": args.w_bit,
        "group_size": args.awq_group_size,
        "zero_point": args.zero_point,
        "asymmetric": asymmetric,
        "preserve_max_abs": args.preserve_max_abs,
        "n_samples": args.awq_n_samples,
        "seqlen": args.awq_seqlen,
        "calib_data": args.calib_data,
        "dtype": args.dtype,
        "total_elapsed_seconds": time.perf_counter() - total_start,
    }
    (output_dir / "awq_pseudo_quantization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
