#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_method(value):
    if "=" not in value:
        raise ValueError(f"Method must be name=/path/to/images, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Invalid method spec: {value}")
    return name, Path(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute paper metrics against a Full SDXL reference directory.")
    parser.add_argument("--reference-dir", type=str, required=True)
    parser.add_argument("--method", action="append", required=True, help="name=/path/to/method/images")
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=512, help="Resize for metric models.")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--dino-model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--disable-clip", action="store_true")
    parser.add_argument("--disable-dino", action="store_true")
    parser.add_argument("--disable-lpips", action="store_true")
    return parser.parse_args()


def image_root(path):
    path = Path(path)
    nested = path / "4_pruned_skip"
    if nested.exists():
        return nested
    return path


def list_images(path):
    root = image_root(path)
    images = sorted([p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise FileNotFoundError(f"No images found in {root}")
    return images


def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def validate_pairs(ref_images, method_images, prompts, method_name):
    if len(ref_images) != len(prompts) or len(method_images) != len(prompts):
        raise ValueError(
            f"{method_name}: expected the same number of reference images, "
            f"method images and prompts; got {len(ref_images)}, "
            f"{len(method_images)} and {len(prompts)}."
        )
    reference_ids = [image.stem for image in ref_images]
    method_ids = [image.stem for image in method_images]
    if len(set(reference_ids)) != len(reference_ids) or len(set(method_ids)) != len(method_ids):
        raise ValueError(f"{method_name}: duplicate image identifiers detected.")
    if reference_ids != method_ids:
        raise ValueError(
            f"{method_name}: reference and method image identifiers do not match; "
            "check the image filenames and ordering before computing paired metrics."
        )
    return len(prompts)


def load_image(path, size=None):
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image


def pil_to_tensor(image, device):
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def psnr(ref, img):
    mse = F.mse_loss(img, ref).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def try_load_lpips(device):
    try:
        import lpips

        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f"LPIPS disabled: {exc}")
        return None


def try_load_clip(model_name, device):
    try:
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()
        return model, processor
    except Exception as exc:
        print(f"CLIP disabled: {exc}")
        return None, None


def try_load_dino(model_name, device):
    try:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        return model, processor
    except Exception as exc:
        print(f"DINO disabled: {exc}")
        return None, None


@torch.no_grad()
def clip_score(model, processor, image, prompt, device):
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return float((image_features * text_features).sum(dim=-1).detach().cpu().item())


@torch.no_grad()
def dino_feature(model, processor, image, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        feat = outputs.pooler_output
    else:
        feat = outputs.last_hidden_state[:, 0]
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def load_summary(path):
    base = Path(path)
    candidates = [base / "obs_acceleration_results.json", base.parent / "obs_acceleration_results.json"]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def mean_or_none(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(np.mean(values))


@torch.no_grad()
def main():
    args = parse_args()
    prompts = load_prompts(args.prompt_file)
    methods = [parse_method(item) for item in args.method]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_images = list_images(args.reference_dir)
    method_images_by_name = {}
    for method_name, method_dir in methods:
        if method_name in method_images_by_name:
            raise ValueError(f"Duplicate method name: {method_name}")
        method_images = list_images(method_dir)
        validate_pairs(ref_images, method_images, prompts, method_name)
        method_images_by_name[method_name] = method_images

    lpips_model = None if args.disable_lpips else try_load_lpips(device)
    clip_model, clip_processor = (None, None) if args.disable_clip else try_load_clip(args.clip_model, device)
    dino_model, dino_processor = (None, None) if args.disable_dino else try_load_dino(args.dino_model, device)
    if not args.disable_lpips and lpips_model is None:
        raise RuntimeError("LPIPS could not be loaded; fix the model or explicitly pass --disable-lpips.")
    if not args.disable_clip and clip_model is None:
        raise RuntimeError("CLIP could not be loaded; fix the model or explicitly pass --disable-clip.")
    if not args.disable_dino and dino_model is None:
        raise RuntimeError("DINO could not be loaded; fix the model or explicitly pass --disable-dino.")

    rows = []
    for method_name, method_dir in methods:
        method_images = method_images_by_name[method_name]
        count = len(prompts)
        if count == 0:
            raise RuntimeError(f"No comparable images for method {method_name}")

        print(f"Computing metrics for {method_name}: {count} images")
        clip_values = []
        lpips_values = []
        dino_values = []
        psnr_values = []

        for idx in range(count):
            prompt = prompts[idx]
            ref_pil = load_image(ref_images[idx], args.image_size)
            img_pil = load_image(method_images[idx], args.image_size)
            ref_tensor = pil_to_tensor(ref_pil, device)
            img_tensor = pil_to_tensor(img_pil, device)

            psnr_values.append(psnr(ref_tensor, img_tensor))

            if lpips_model is not None:
                ref_lpips = ref_tensor * 2.0 - 1.0
                img_lpips = img_tensor * 2.0 - 1.0
                value = lpips_model(img_lpips, ref_lpips).mean().detach().cpu().item()
                lpips_values.append(float(value))

            if clip_model is not None:
                clip_values.append(clip_score(clip_model, clip_processor, img_pil, prompt, device))

            if dino_model is not None:
                ref_feat = dino_feature(dino_model, dino_processor, ref_pil, device)
                img_feat = dino_feature(dino_model, dino_processor, img_pil, device)
                dist = 1.0 - (ref_feat * img_feat).sum(dim=-1)
                dino_values.append(float(dist.detach().cpu().item()))

        summary = load_summary(method_dir)
        rows.append(
            {
                "method": method_name,
                "num_images": count,
                "avg_time": summary.get("pruned_skip_avg_time"),
                "speedup": summary.get("pruned_skip_speedup"),
                "skip_rate": summary.get("pruned_skip_rate"),
                "clip_score": mean_or_none(clip_values),
                "lpips_to_full": mean_or_none(lpips_values),
                "dino_distance_to_full": mean_or_none(dino_values),
                "psnr_to_full": mean_or_none(psnr_values),
                "image_dir": str(method_dir),
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metrics: {output_csv}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
