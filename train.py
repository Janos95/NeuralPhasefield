import argparse
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


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    dataset = BlenderDataset(args.dataset_root, split="train")
    model = PhaseFieldModel().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume is not None:
        ckpt_path = Path(args.resume)
        start_step = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resumed from {ckpt_path} at step {start_step}")

    model.train()

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
        loss = img_loss + args.lambda_mm * mm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print(f"step {step:06d} | loss {loss.item():.5f} | img {img_loss.item():.5f} | mm {mm_loss.item():.5f}")

        if step % args.psnr_interval == 0:
            with torch.no_grad():
                mse = img_loss.item()
                psnr = -10.0 * torch.log10(torch.tensor(mse + 1e-10)).item()
            print(f"step {step:06d} | psnr {psnr:.2f} dB")

        if step % args.save_interval == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"step_{step:06d}.pt"
            save_checkpoint(ckpt_path, step, model, optimizer)

    final_ckpt = Path(args.checkpoint_dir) / "latest.pt"
    save_checkpoint(final_ckpt, args.max_iters, model, optimizer)
    print(f"Training complete. Saved final checkpoint to {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Phase-Field Radiance Field training")
    parser.add_argument("--dataset-root", default="data/nerf_synthetic/lego", help="Root to Blender LEGO dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rays-per-batch", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--eps-rend", type=float, default=0.01)
    parser.add_argument("--eps-pen", type=float, default=0.02)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--lambda-mm", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-iters", type=int, default=200_000)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--psnr-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
