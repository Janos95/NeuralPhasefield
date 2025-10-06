import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F

from data_blender import BlenderDataset
from models import PhaseFieldModel, integrate_rays


def save_checkpoint(path: Path, step: int, model: nn.Module, optimizer: torch.optim.Optimizer):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer]):
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model"])
    if optimizer is not None and "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
    return data.get("step", 0)


def apply_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """Overwrite argparse namespace values with entries from a JSON config."""
    if args.config is None:
        return args

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    if not isinstance(config_data, dict):
        raise ValueError("Config file must contain a JSON object at the top level.")

    defaults = {action.dest: action.default for action in parser._actions if action.dest is not None}

    for key, value in config_data.items():
        if key == "config":
            continue  # ignore recursive specification
        if not hasattr(args, key):
            raise KeyError(f"Unknown config key '{key}'")

        current = getattr(args, key)
        default = defaults.get(key)
        if key == "train_frame_indices" and isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)

        if current == default:
            setattr(args, key, value)

    return args


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    frame_subset = None
    if args.train_frame_indices:
        frame_subset = [int(item.strip()) for item in args.train_frame_indices.split(",") if item.strip()]

    dataset = BlenderDataset(args.dataset_root, split="train", frame_subset=frame_subset)
    if not (0 <= args.model_skip_layer < args.model_num_trunk_layers):
        raise ValueError("model_skip_layer must lie in [0, model_num_trunk_layers).")

    model = PhaseFieldModel(
        pos_freqs=args.model_pos_freqs,
        dir_freqs=args.model_dir_freqs,
        hidden_dim=args.model_hidden_dim,
        num_trunk_layers=args.model_num_trunk_layers,
        skip_layer=args.model_skip_layer,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as err:  # pragma: no cover - optional dependency guard
            raise ImportError("Weights & Biases is not installed. Install it or omit --wandb.") from err

        tags = None
        if args.wandb_tags:
            tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]

        wandb_init_kwargs = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_run_name,
            "group": args.wandb_group,
            "tags": tags,
            "mode": args.wandb_mode,
            "config": {
                "dataset_root": args.dataset_root,
                "rays_per_batch": args.rays_per_batch,
                "num_samples": args.num_samples,
                "near": args.near,
                "far": args.far,
                "eps_rend": args.eps_rend,
                "eps_pen": args.eps_pen,
                "kappa": args.kappa,
                "lambda_mm": args.lambda_mm,
                "lr": args.lr,
                "seed": args.seed,
                "device": args.device,
                "max_iters": args.max_iters,
                "train_frame_indices": frame_subset,
            },
        }

        # Drop keys with None values to keep the call clean.
        wandb_init_kwargs = {k: v for k, v in wandb_init_kwargs.items() if v is not None}
        wandb_run = wandb.init(**wandb_init_kwargs)

    start_step = 0
    if args.resume is not None:
        ckpt_path = Path(args.resume)
        start_step = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resumed from {ckpt_path} at step {start_step}")

    model.train()
    last_psnr = None

    for step in range(start_step + 1, args.max_iters + 1):
        ray_origins, ray_dirs, target_rgb = dataset.sample_random_rays(args.rays_per_batch, device)
        outputs = integrate_rays(
            model,
            ray_origins,
            ray_dirs,
            num_samples=args.num_samples,
            near=args.near,
            far=args.far,
            eps_rend=args.eps_rend,
            kappa=args.kappa,
            create_graph=True,
        )

        pred_rgb = outputs["rgb"]
        phi_samples = outputs["phi"]
        grad_samples = outputs["grads"]

        img_loss = F.mse_loss(pred_rgb, target_rgb)
        grad_norm_sq = torch.sum(grad_samples ** 2, dim=-1)
        mm_penalty = args.eps_pen * grad_norm_sq + (1.0 / args.eps_pen) * (phi_samples ** 2 - 1.0) ** 2
        mm_loss = mm_penalty.mean()
        lambda_mm_eff = args.lambda_mm
        if args.lambda_mm_ramp_iters > 0:
            ramp = min(1.0, step / args.lambda_mm_ramp_iters)
            lambda_mm_eff = lambda_mm_eff * ramp

        loss = img_loss + lambda_mm_eff * mm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            total_loss = loss.item()
            img_val = img_loss.item()
            mm_val = mm_loss.item()
            print(f"step {step:06d} | loss {total_loss:.5f} | img {img_val:.5f} | mm {mm_val:.5f}")
            if wandb_run is not None:
                wandb.log({
                    "loss/total": total_loss,
                    "loss/img": img_val,
                    "loss/mm": mm_val,
                    "loss/lambda_mm_eff": lambda_mm_eff,
                }, step=step)

        if step % args.psnr_interval == 0:
            with torch.no_grad():
                mse = img_loss.item()
                psnr = -10.0 * torch.log10(torch.tensor(mse + 1e-10)).item()
            print(f"step {step:06d} | psnr {psnr:.2f} dB")
            last_psnr = psnr
            if wandb_run is not None:
                wandb.log({"metrics/psnr": psnr}, step=step)

        if step % args.save_interval == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"step_{step:06d}.pt"
            save_checkpoint(ckpt_path, step, model, optimizer)

    final_ckpt = Path(args.checkpoint_dir) / "latest.pt"
    save_checkpoint(final_ckpt, args.max_iters, model, optimizer)
    print(f"Training complete. Saved final checkpoint to {final_ckpt}")
    if wandb_run is not None:
        if last_psnr is not None:
            wandb.log({"metrics/final_psnr": last_psnr}, step=args.max_iters)
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Phase-Field Radiance Field training")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config with training arguments.")
    parser.add_argument("--dataset-root", default="data/nerf_synthetic/lego", help="Root to Blender LEGO dataset")
    parser.add_argument("--model-pos-freqs", type=int, default=10, help="Number of position encoding frequencies.")
    parser.add_argument("--model-dir-freqs", type=int, default=4, help="Number of direction encoding frequencies.")
    parser.add_argument("--model-hidden-dim", type=int, default=256, help="Hidden layer width for the trunk MLP.")
    parser.add_argument("--model-num-trunk-layers", type=int, default=8, help="Number of layers in the shared trunk.")
    parser.add_argument("--model-skip-layer", type=int, default=4, help="Index of the skip connection layer in the trunk.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rays-per-batch", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--eps-rend", type=float, default=0.01)
    parser.add_argument("--eps-pen", type=float, default=0.02)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--lambda-mm", type=float, default=1e-2)
    parser.add_argument(
        "--lambda-mm-ramp-iters",
        type=int,
        default=0,
        help="Linearly ramp lambda_mm from 0 to its target value over this many steps.",
    )
    parser.add_argument(
        "--train-frame-indices",
        type=str,
        default=None,
        help="Comma-separated list of frame indices to train on (e.g. '0' to overfit a single image).",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-iters", type=int, default=200_000)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--psnr-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="phasefield-nerf")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated list of WandB tags")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"], help="WandB operating mode")
    args = parser.parse_args()
    args = apply_config(parser, args)
    train(args)
