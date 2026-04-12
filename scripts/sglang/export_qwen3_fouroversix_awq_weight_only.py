from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from awq.quantize.pre_quant import apply_awq, run_awq
from fouroversix import (
    FourOverSixWeightOnlyLinear,
    ModelQuantizationConfig,
    apply_offline_weight_quantization,
    quantize_model,
)
from fouroversix.model.quantize import QuantizedModule
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.sglang.modeling_qwen3_fouroversix_awq_weight_only import patch_config_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--awq-cache-path", default=None)
    parser.add_argument("--awq-group-size", type=int, default=-1)
    parser.add_argument("--awq-n-samples", type=int, default=128)
    parser.add_argument("--awq-seqlen", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def default_awq_cache_path(
    model_name: str,
    *,
    awq_group_size: int,
    awq_n_samples: int,
    awq_seqlen: int,
) -> Path:
    model_slug = model_name.replace("/", "__")
    return Path("ptq/awq_weight_only") / model_slug / (
        f"g{awq_group_size}_n{awq_n_samples}_s{awq_seqlen}.pt"
    )


def load_or_run_awq(
    model_name: str,
    cache_path: Path,
    *,
    device: str,
    trust_remote_code: bool,
    quantization_config: ModelQuantizationConfig,
    awq_group_size: int,
    awq_n_samples: int,
    awq_seqlen: int,
) -> dict:
    if cache_path.exists():
        return torch.load(cache_path, map_location=device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=trust_remote_code,
    ).eval()
    quantize_model(model, quantization_config)

    enc = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=trust_remote_code,
    )

    awq_results = run_awq(
        model,
        enc,
        w_bit=16,
        q_config={"q_group_size": awq_group_size, "zero_point": False},
        n_samples=awq_n_samples,
        seqlen=awq_seqlen,
        calib_data="wikitext",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(awq_results, cache_path)
    return awq_results


def _jsonify(value):
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.force:
            msg = f"{output_dir} already exists; pass --force to overwrite."
            raise FileExistsError(msg)
        shutil.rmtree(output_dir)

    quantization_config = ModelQuantizationConfig()
    awq_cache_path = (
        Path(args.awq_cache_path)
        if args.awq_cache_path is not None
        else default_awq_cache_path(
            args.model_name,
            awq_group_size=args.awq_group_size,
            awq_n_samples=args.awq_n_samples,
            awq_seqlen=args.awq_seqlen,
        )
    )
    awq_results = load_or_run_awq(
        args.model_name,
        awq_cache_path,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quantization_config,
        awq_group_size=args.awq_group_size,
        awq_n_samples=args.awq_n_samples,
        awq_seqlen=args.awq_seqlen,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    apply_awq(model, awq_results)

    QuantizedModule.register(
        nn.Linear,
        replace_existing_modules_in_registry=True,
    )(FourOverSixWeightOnlyLinear)
    quantize_model(model, quantization_config)
    apply_offline_weight_quantization(model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    template_path = (
        Path(__file__).resolve().parent / "modeling_qwen3_fouroversix_awq_weight_only.py"
    )
    shutil.copy2(
        template_path,
        output_dir / "modeling_qwen3_fouroversix_awq_weight_only.py",
    )
    patch_config_json(output_dir)

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["fouroversix_awq_weight_only_config"] = _jsonify(
        quantization_config.__dict__
    )
    config["awq_weight_only_config"] = {
        "q_group_size": args.awq_group_size,
        "zero_point": False,
        "n_samples": args.awq_n_samples,
        "seqlen": args.awq_seqlen,
    }
    config.pop("quantization_config", None)
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
