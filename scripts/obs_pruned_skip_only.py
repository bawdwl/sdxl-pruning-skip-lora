#!/usr/bin/env python3
import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, DiffusionPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive_diffusion"))

from acceleration.sparse_pipeline import StableDiffusionXLPipeline as AdaptiveStableDiffusionXLPipeline


DEFAULT_PROMPTS = [
    "A cozy library with a roaring fireplace",
    "A magical tree with glowing fruit in a dark forest",
    "A mystical wolf howling under a glowing aurora",
    "A panda playing the guitar under a cherry blossom tree",
    "A robot dog exploring an abandoned spaceship",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--pruned-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--risk-table",
        type=str,
        default=None,
    )
    parser.add_argument("--risk-weight", type=float, default=0.0)
    parser.add_argument("--repaired-skip-risk-table", type=str, default=None)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--max-skip-steps", type=int, default=4)
    parser.add_argument("--late-protection-ratio", type=float, default=0.0)
    parser.add_argument("--late-threshold-scale", type=float, default=1.0)
    parser.add_argument("--skip-extrapolation-strength", type=float, default=0.0)
    parser.add_argument("--skip-extrapolation-max-ratio", type=float, default=0.15)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--baseline-time", type=float, default=None)
    return parser.parse_args()


def load_prompts(args):
    if args.prompt_file is None:
        return DEFAULT_PROMPTS[: args.num_prompts]

    prompts = []
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts[: args.num_prompts]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pruned_unet(path, device, dtype):
    suffix = Path(path).suffix.lower()
    if suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f).to(device=device, dtype=dtype)
    return torch.load(path, weights_only=False).to(device=device, dtype=dtype)


def get_skip_stats(adaptive_pipe):
    if not hasattr(adaptive_pipe, "mask") or not adaptive_pipe.mask:
        return {
            "total": 0,
            "recomputed": 0,
            "skipped": 0,
            "recompute_rate": 0.0,
            "skip_rate": 0.0,
        }

    mask = adaptive_pipe.mask
    total = len(mask)
    recomputed = sum(mask)
    skipped = total - recomputed
    return {
        "total": total,
        "recomputed": recomputed,
        "skipped": skipped,
        "recompute_rate": recomputed / total * 100,
        "skip_rate": skipped / total * 100,
    }


@torch.no_grad()
def main():
    args = parse_args()
    prompts = load_prompts(args)
    if not prompts:
        raise ValueError("No prompts to render.")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "4_pruned_skip"
    image_dir.mkdir(parents=True, exist_ok=True)

    risk_table_for_pipeline = args.risk_table if args.risk_weight != 0 else None

    print("=" * 80)
    print("OBS pruned + adaptive skip only")
    print("=" * 80)
    print(f"Output dir: {output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Late protection ratio: {args.late_protection_ratio}")
    print(f"Late threshold scale: {args.late_threshold_scale}")
    print(f"Skip extrapolation strength: {args.skip_extrapolation_strength}")
    print(f"Skip extrapolation max ratio: {args.skip_extrapolation_max_ratio}")
    print(f"Risk weight: {args.risk_weight}")
    print(f"Risk table: {risk_table_for_pipeline if risk_table_for_pipeline else 'disabled'}")
    print(f"Repaired skip risk table: {args.repaired_skip_risk_table if args.repaired_skip_risk_table else 'disabled'}")
    print(f"LoRA path: {args.lora_path if args.lora_path else 'disabled'}")
    print(f"Prompts: {len(prompts)}")

    set_seed(args.seed)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(device)
    base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)
    base_pipe.unet = load_pruned_unet(args.pruned_path, device, dtype)
    if args.lora_path:
        print("Loading LoRA weights...")
        base_pipe.load_lora_weights(args.lora_path)
        base_pipe.fuse_lora(lora_scale=args.lora_scale)

    adaptive_pipe = AdaptiveStableDiffusionXLPipeline.from_pipe(
        base_pipe,
        threshold=args.threshold,
        max_skip_steps=args.max_skip_steps,
        pruning_risk_table=risk_table_for_pipeline,
        pruning_risk_weight=args.risk_weight,
        repaired_skip_risk_table=args.repaired_skip_risk_table,
        late_protection_ratio=args.late_protection_ratio,
        late_threshold_scale=args.late_threshold_scale,
        skip_extrapolation_strength=args.skip_extrapolation_strength,
        skip_extrapolation_max_ratio=args.skip_extrapolation_max_ratio,
    )

    print("Warmup...")
    _ = adaptive_pipe("warmup", num_inference_steps=10, height=args.height, width=args.width)
    adaptive_pipe.reset_cache()

    times = []
    skip_rates = []
    per_prompt = []
    extrapolation_calls = []

    for i, prompt in enumerate(prompts):
        set_seed(args.seed + i)
        print(f"[{i + 1}/{len(prompts)}] {prompt[:80]}")
        start = time.time()
        output = adaptive_pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
        )
        elapsed = time.time() - start
        stats = get_skip_stats(adaptive_pipe)
        prompt_extrapolation_calls = getattr(adaptive_pipe, "skip_extrapolation_calls", 0)

        output.images[0].save(image_dir / f"{i:02d}.png")
        times.append(elapsed)
        skip_rates.append(stats["skip_rate"])
        extrapolation_calls.append(prompt_extrapolation_calls)
        per_prompt.append(
            {
                "prompt": prompt,
                "time": elapsed,
                "skip_extrapolation_calls": prompt_extrapolation_calls,
                **stats,
            }
        )

        print(
            f"  time={elapsed:.2f}s "
            f"recomputed={stats['recomputed']}/{stats['total']} ({stats['recompute_rate']:.1f}%) "
            f"skipped={stats['skipped']}/{stats['total']} ({stats['skip_rate']:.1f}%)"
        )
        adaptive_pipe.reset_cache()

    avg_time = float(np.mean(times))
    avg_skip_rate = float(np.mean(skip_rates))
    speedup = None
    if args.baseline_time is not None:
        speedup = float((args.baseline_time - avg_time) / args.baseline_time * 100)

    summary = {
        "results_dir": str(output_dir),
        "model_path": args.model_path,
        "pruned_path": args.pruned_path,
        "risk_table": risk_table_for_pipeline,
        "risk_weight": args.risk_weight,
        "repaired_skip_risk_table": args.repaired_skip_risk_table,
        "lora_path": args.lora_path,
        "lora_scale": args.lora_scale,
        "threshold": args.threshold,
        "max_skip_steps": args.max_skip_steps,
        "late_protection_ratio": args.late_protection_ratio,
        "late_threshold_scale": args.late_threshold_scale,
        "skip_extrapolation_strength": args.skip_extrapolation_strength,
        "skip_extrapolation_max_ratio": args.skip_extrapolation_max_ratio,
        "skip_extrapolation_calls": int(sum(extrapolation_calls)),
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "pruned_skip_avg_time": avg_time,
        "pruned_skip_rate": avg_skip_rate,
        "pruned_skip_speedup": speedup,
        "per_prompt": per_prompt,
    }

    with open(output_dir / "obs_acceleration_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / "obs_acceleration_results.txt", "w", encoding="utf-8") as f:
        f.write("Experiment, Avg time(s), Real skip rate\n")
        f.write(f"OBS_Pruned_Skip, {avg_time:.2f}, {avg_skip_rate:.1f}%\n")

    del adaptive_pipe, base_pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 80)
    print(f"Avg time: {avg_time:.2f}s")
    print(f"Avg real skip rate: {avg_skip_rate:.1f}%")
    if speedup is not None:
        print(f"Speedup vs baseline: {speedup:+.1f}%")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
