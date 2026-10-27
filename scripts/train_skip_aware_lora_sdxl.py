#!/usr/bin/env python3
import argparse
import gc
import inspect
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict


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
        "--output-dir",
        type=str,
        required=True,
    )
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--validation-prompt-file", type=str, default=None)
    parser.add_argument("--num-validation-prompts", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--max-skip-steps", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument(
        "--init-lora-path",
        type=str,
        default=None,
        help=(
            "Optional LoRA directory used to initialize the student before collecting skip segments. "
            "Use this for self-consistent trajectory training."
        ),
    )
    parser.add_argument("--max-segments-per-prompt", type=int, default=6)
    parser.add_argument("--min-skip-length", type=int, default=1)
    parser.add_argument(
        "--segment-weight-mode",
        choices=["none", "skip_length", "late_stage"],
        default="none",
        help="Optional weighting for unrolled skip segments. Keep none for the original method.",
    )
    parser.add_argument(
        "--skip-length-weight",
        type=float,
        default=0.25,
        help="Alpha for skip_length weighting: w = 1 + alpha * number_of_skipped_steps.",
    )
    parser.add_argument(
        "--late-stage-weight",
        type=float,
        default=0.5,
        help="Beta for late_stage weighting: w = 1 + beta * anchor_step_ratio.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=["unrolled", "anchor_eps"],
        default="unrolled",
        help=(
            "unrolled: skip-aware loss on the latent after reusing the anchor prediction. "
            "anchor_eps: ordinary LoRA baseline that only distills the teacher epsilon at the anchor step."
        ),
    )
    parser.add_argument(
        "--unrolled-loss-mode",
        choices=["final", "intermediate"],
        default="final",
        help=(
            "final: original skip-aware loss, only compares the latent at the end of a skip segment. "
            "intermediate: compares every reused-noise latent inside the skip segment and gives the final latent extra weight."
        ),
    )
    parser.add_argument(
        "--intermediate-final-weight",
        type=float,
        default=1.0,
        help="Extra weight for the final latent when --unrolled-loss-mode intermediate is used.",
    )
    parser.add_argument("--anchor-eps-weight", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable UNet gradient checkpointing. Useful for 1024x1024 LoRA continuation on 24GB GPUs.",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        default="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2",
    )
    return parser.parse_args()


def load_prompt_file(path, limit=None):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts[:limit] if limit else prompts


def load_prompts(args):
    if args.prompt_file is None:
        return DEFAULT_PROMPTS[: args.num_prompts]
    return load_prompt_file(args.prompt_file, args.num_prompts)


def load_validation_prompts(args):
    if args.validation_prompt_file is None:
        return []
    limit = args.num_validation_prompts if args.num_validation_prompts > 0 else None
    return load_prompt_file(args.validation_prompt_file, limit)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_pruned_unet(path, device, dtype):
    suffix = Path(path).suffix.lower()
    if suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f).to(device=device, dtype=dtype)
    return torch.load(path, weights_only=False).to(device=device, dtype=dtype)


def load_lora_into_unet(unet, lora_path):
    lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_path)
    unet_state_dict = {
        key.replace("unet.", ""): value
        for key, value in lora_state_dict.items()
        if key.startswith("unet.")
    }
    unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
    incompatible_keys = set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")
    if incompatible_keys is not None:
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            print(f"Warning: unexpected LoRA keys while loading {lora_path}: {unexpected_keys}")


def get_add_time_ids(pipe, height, width, dtype, device):
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    kwargs = {"dtype": dtype}
    signature = inspect.signature(pipe._get_add_time_ids)
    if "text_encoder_projection_dim" in signature.parameters:
        kwargs["text_encoder_projection_dim"] = pipe.text_encoder_2.config.projection_dim
    add_time_ids = pipe._get_add_time_ids(original_size, crops_coords_top_left, target_size, **kwargs)
    return add_time_ids.to(device=device, dtype=dtype)


def encode_prompt_context(pipe, prompt, args, dtype, device):
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device=device, dtype=dtype)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0).to(
        device=device, dtype=dtype
    )
    add_time_ids = get_add_time_ids(pipe, args.height, args.width, dtype, device)
    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(device=device, dtype=dtype)
    return {
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "add_text_embeds": add_text_embeds.detach().cpu(),
        "add_time_ids": add_time_ids.detach().cpu(),
    }


def context_to_device(ctx, device, dtype):
    return {
        "prompt_embeds": ctx["prompt_embeds"].to(device=device, dtype=dtype),
        "add_text_embeds": ctx["add_text_embeds"].to(device=device, dtype=dtype),
        "add_time_ids": ctx["add_time_ids"].to(device=device, dtype=dtype),
    }


def combine_cfg(noise_pred, guidance_scale):
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


def run_unet_guided(unet, pipe, latents, timestep, ctx, guidance_scale):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
    added_cond_kwargs = {"text_embeds": ctx["add_text_embeds"], "time_ids": ctx["add_time_ids"]}

    timestep_cond = None
    if getattr(unet.config, "time_cond_proj_dim", None) is not None:
        guidance_scale_tensor = torch.tensor(guidance_scale - 1, device=latents.device).repeat(latents.shape[0])
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=unet.config.time_cond_proj_dim,
        ).to(device=latents.device, dtype=latents.dtype)

    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=ctx["prompt_embeds"],
        timestep_cond=timestep_cond,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
    return combine_cfg(noise_pred, guidance_scale)


def build_compute_flags(mask, num_steps):
    flags = [True] * num_steps
    for step_idx in range(4, num_steps):
        mask_idx = step_idx - 4
        flags[step_idx] = bool(mask[mask_idx]) if mask_idx < len(mask) else True
    return flags


def find_skip_segments(flags, min_skip_length):
    segments = []
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i + 1
        while j < len(flags) and not flags[j]:
            j += 1
        skip_length = j - i - 1
        if skip_length >= min_skip_length:
            segments.append((i, j))
        i = j
    return segments


@torch.no_grad()
def collect_prompt_schedule(adaptive_pipe, prompt, args, seed):
    set_seed(seed)
    _ = adaptive_pipe(
        prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    )
    mask = list(adaptive_pipe.mask)
    adaptive_pipe.reset_cache()
    flags = build_compute_flags(mask, args.num_inference_steps)
    return mask, flags


@torch.no_grad()
def collect_segment_samples(pipe, pruned_unet, prompt, ctx_cpu, flags, segments, args, seed, dtype, device):
    ctx = context_to_device(ctx_cpu, device, dtype)
    timesteps = pipe.scheduler.timesteps
    anchor_to_end = {start: end for start, end in segments}

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pruned_unet.config.in_channels,
        height=args.height,
        width=args.width,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,
    )

    samples = []
    cached_noise = None
    for step_idx, timestep in enumerate(timesteps):
        if step_idx in anchor_to_end:
            samples.append(
                {
                    "prompt": prompt,
                    "ctx": ctx_cpu,
                    "anchor_step": step_idx,
                    "end_step": anchor_to_end[step_idx],
                    "latents": latents.detach().cpu(),
                }
            )

        if flags[step_idx] or cached_noise is None:
            cached_noise = run_unet_guided(pruned_unet, pipe, latents, timestep, ctx, args.guidance_scale)
        latents = pipe.scheduler.step(cached_noise, timestep, latents, return_dict=False)[0]

    return samples


def segment_loss(pipe, teacher_unet, student_unet, sample, args, timesteps, dtype, device, train):
    ctx = context_to_device(sample["ctx"], device, dtype)
    latents = sample["latents"].to(device=device, dtype=dtype)
    start = sample["anchor_step"]
    end = sample["end_step"]

    if train:
        student_noise = run_unet_guided(student_unet, pipe, latents, timesteps[start], ctx, args.guidance_scale)
    else:
        with torch.no_grad():
            student_noise = run_unet_guided(student_unet, pipe, latents, timesteps[start], ctx, args.guidance_scale)

    with torch.no_grad():
        teacher_anchor_noise = run_unet_guided(teacher_unet, pipe, latents, timesteps[start], ctx, args.guidance_scale)

    student_latents = latents
    teacher_latents = latents.detach()
    latent_terms = []

    for step_idx in range(start, end):
        student_latents = pipe.scheduler.step(
            student_noise,
            timesteps[step_idx],
            student_latents,
            return_dict=False,
        )[0]

        with torch.no_grad():
            teacher_noise = run_unet_guided(
                teacher_unet,
                pipe,
                teacher_latents,
                timesteps[step_idx],
                ctx,
                args.guidance_scale,
            )
            teacher_latents = pipe.scheduler.step(
                teacher_noise,
                timesteps[step_idx],
                teacher_latents,
                return_dict=False,
            )[0]

        if args.loss_mode == "unrolled" and args.unrolled_loss_mode == "intermediate":
            latent_terms.append(F.mse_loss(student_latents.float(), teacher_latents.float()))

    final_latent_loss = F.mse_loss(student_latents.float(), teacher_latents.float())
    if args.loss_mode == "unrolled" and args.unrolled_loss_mode == "intermediate" and latent_terms:
        latent_terms.append(args.intermediate_final_weight * final_latent_loss)
        latent_loss = torch.stack(latent_terms).sum() / (len(latent_terms) - 1 + args.intermediate_final_weight)
    else:
        latent_loss = final_latent_loss
    eps_loss = F.mse_loss(student_noise.float(), teacher_anchor_noise.float())
    if args.loss_mode == "anchor_eps":
        loss = eps_loss
    else:
        skipped_steps = max(0, end - start - 1)
        segment_weight = 1.0
        if args.segment_weight_mode == "skip_length":
            segment_weight = 1.0 + args.skip_length_weight * skipped_steps
        elif args.segment_weight_mode == "late_stage":
            anchor_ratio = start / max(len(timesteps) - 1, 1)
            segment_weight = 1.0 + args.late_stage_weight * anchor_ratio
        loss = segment_weight * latent_loss + args.anchor_eps_weight * eps_loss
    return loss, latent_loss.detach(), eps_loss.detach()


def evaluate_samples(pipe, teacher_unet, student_unet, samples, args, timesteps, dtype, device, max_items=None):
    if not samples:
        return {"loss": None, "latent_loss": None, "eps_loss": None}
    student_unet.eval()
    total_loss = 0.0
    total_latent = 0.0
    total_eps = 0.0
    count = 0
    subset = samples[:max_items] if max_items is not None else samples
    for sample in subset:
        loss, latent_loss, eps_loss = segment_loss(
            pipe, teacher_unet, student_unet, sample, args, timesteps, dtype, device, train=False
        )
        total_loss += float(loss.detach().cpu())
        total_latent += float(latent_loss.detach().cpu())
        total_eps += float(eps_loss.detach().cpu())
        count += 1
    student_unet.train()
    return {
        "loss": total_loss / count,
        "latent_loss": total_latent / count,
        "eps_loss": total_eps / count,
    }


@torch.no_grad()
def collect_samples_for_prompts(pipe, adaptive_pipe, student_unet, prompts, args, dtype, device, seed_offset=0):
    all_samples = []
    schedule_payload = []
    for prompt_idx, prompt in enumerate(prompts):
        prompt_seed = args.seed + seed_offset + prompt_idx
        ctx_cpu = encode_prompt_context(pipe, prompt, args, dtype, device)
        mask, flags = collect_prompt_schedule(adaptive_pipe, prompt, args, prompt_seed)
        segments = find_skip_segments(flags, args.min_skip_length)
        segments = sorted(segments, key=lambda item: item[1] - item[0], reverse=True)
        segments = segments[: args.max_segments_per_prompt]
        segments = sorted(segments)
        prompt_samples = collect_segment_samples(
            pipe,
            student_unet,
            prompt,
            ctx_cpu,
            flags,
            segments,
            args,
            prompt_seed,
            dtype,
            device,
        )
        all_samples.extend(prompt_samples)
        schedule_payload.append(
            {
                "prompt": prompt,
                "mask": mask,
                "compute_flags": flags,
                "segments": [{"start": start, "end": end, "length": end - start} for start, end in segments],
            }
        )
        print(f"  [{prompt_idx + 1}/{len(prompts)}] segments={len(prompt_samples)} mask_len={len(mask)}")
    return all_samples, schedule_payload


def main():
    args = parse_args()
    train_prompts = load_prompts(args)
    validation_prompts = load_validation_prompts(args)
    if not train_prompts:
        raise ValueError("No prompts were provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("=" * 80)
    print("Skip-aware LoRA repair with unrolled skip loss")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Pruned path: {args.pruned_path}")
    print(f"Output dir: {output_dir}")
    print(f"Train prompts: {len(train_prompts)}")
    print(f"Validation prompts: {len(validation_prompts)}")
    print(f"Steps: {args.num_inference_steps}, size: {args.height}x{args.width}")
    print(f"Threshold: {args.threshold}, max skip steps: {args.max_skip_steps}")
    print(f"Loss mode: {args.loss_mode}")
    print(f"Segment weight mode: {args.segment_weight_mode}")
    print(f"Skip length weight: {args.skip_length_weight}")
    print(f"Late stage weight: {args.late_stage_weight}")
    print(f"Train steps: {args.train_steps}, rank: {args.rank}, lr: {args.learning_rate}")
    print(f"Init LoRA path: {args.init_lora_path if args.init_lora_path else 'disabled'}")

    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    teacher_unet = pipe.unet
    teacher_unet.eval()
    teacher_unet.requires_grad_(False)

    print("Loading pruned student UNet...")
    student_unet = load_pruned_unet(args.pruned_path, device, dtype)
    student_unet.eval()
    student_unet.requires_grad_(False)
    pipe.unet = student_unet

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    student_unet.add_adapter(lora_config)
    if args.init_lora_path:
        print("Loading initial LoRA before self-consistent segment collection...")
        load_lora_into_unet(student_unet, args.init_lora_path)
    if args.gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()
    cast_training_params(student_unet, dtype=torch.float32)
    student_unet.eval()

    adaptive_pipe = AdaptiveStableDiffusionXLPipeline.from_pipe(
        pipe,
        threshold=args.threshold,
        max_skip_steps=args.max_skip_steps,
    )

    print("Collecting train adaptive skip segments...")
    all_samples, schedule_payload = collect_samples_for_prompts(
        pipe, adaptive_pipe, student_unet, train_prompts, args, dtype, device, seed_offset=0
    )

    val_samples = []
    validation_schedule_payload = []
    if validation_prompts:
        print("Collecting validation adaptive skip segments...")
        val_samples, validation_schedule_payload = collect_samples_for_prompts(
            pipe,
            adaptive_pipe,
            student_unet,
            validation_prompts,
            args,
            dtype,
            device,
            seed_offset=10000,
        )

    del adaptive_pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_samples:
        raise RuntimeError("No skip segments were collected. Try a higher threshold or lower min-skip-length.")

    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    if validation_prompts:
        train_samples = all_samples
    else:
        val_count = max(1, int(len(all_samples) * 0.2)) if len(all_samples) > 1 else 0
        val_samples = all_samples[:val_count]
        train_samples = all_samples[val_count:] if val_count else all_samples

    with open(output_dir / "skip_segments.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": schedule_payload,
                "validation": validation_schedule_payload,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Training segments: {len(train_samples)}, validation segments: {len(val_samples)}")

    student_unet.train()

    trainable_params = [p for p in student_unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters were found.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    initial_train_metrics = evaluate_samples(
        pipe, teacher_unet, student_unet, train_samples, args, timesteps, dtype, device, max_items=min(8, len(train_samples))
    )
    initial_val_metrics = evaluate_samples(
        pipe, teacher_unet, student_unet, val_samples, args, timesteps, dtype, device
    )
    print(f"Initial train loss: {initial_train_metrics}")
    print(f"Initial val loss: {initial_val_metrics}")

    history = []
    for step in range(1, args.train_steps + 1):
        sample = rng.choice(train_samples)
        optimizer.zero_grad(set_to_none=True)
        loss, latent_loss, eps_loss = segment_loss(
            pipe, teacher_unet, student_unet, sample, args, timesteps, dtype, device, train=True
        )
        loss.backward()
        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()

        if step == 1 or step % 10 == 0 or step == args.train_steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "latent_loss": float(latent_loss.detach().cpu()),
                "eps_loss": float(eps_loss.detach().cpu()),
            }
            history.append(row)
            print(
                f"step={step:04d} "
                f"loss={row['loss']:.6f} "
                f"latent={row['latent_loss']:.6f} "
                f"eps={row['eps_loss']:.6f}"
            )

    final_train_metrics = evaluate_samples(
        pipe, teacher_unet, student_unet, train_samples, args, timesteps, dtype, device, max_items=min(8, len(train_samples))
    )
    final_val_metrics = evaluate_samples(
        pipe, teacher_unet, student_unet, val_samples, args, timesteps, dtype, device
    )
    print(f"Final train loss: {final_train_metrics}")
    print(f"Final val loss: {final_val_metrics}")

    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(student_unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=None,
        text_encoder_2_lora_layers=None,
    )

    summary = {
        "model_path": args.model_path,
        "pruned_path": args.pruned_path,
        "output_dir": str(output_dir),
        "num_prompts": len(train_prompts),
        "num_validation_prompts": len(validation_prompts),
        "train_prompts": train_prompts,
        "validation_prompts": validation_prompts,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "threshold": args.threshold,
        "max_skip_steps": args.max_skip_steps,
        "init_lora_path": args.init_lora_path,
        "loss_mode": args.loss_mode,
        "unrolled_loss_mode": args.unrolled_loss_mode,
        "intermediate_final_weight": args.intermediate_final_weight,
        "segment_weight_mode": args.segment_weight_mode,
        "skip_length_weight": args.skip_length_weight,
        "late_stage_weight": args.late_stage_weight,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "train_steps": args.train_steps,
        "num_train_segments": len(train_samples),
        "num_val_segments": len(val_samples),
        "initial_train_metrics": initial_train_metrics,
        "initial_val_metrics": initial_val_metrics,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "history": history,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Saved LoRA and summary to: {output_dir}")
    print("Next: evaluate this directory with obs_pruned_skip_only.py --lora-path")


if __name__ == "__main__":
    main()
