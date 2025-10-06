import argparse
import os
from pathlib import Path

import imageio.v3 as iio
import torch
from torch.nn import functional as F

from data_blender import BlenderDataset
from models import PhaseFieldModel, integrate_rays


def render_image(args):
    device = torch.device(args.device)
    dataset = BlenderDataset(args.dataset_root, split=args.pose_split)
    pose = dataset.get_pose(args.pose_index)

    model = PhaseFieldModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    origins, dirs = dataset.generate_rays_for_pose(pose, device)
    total_rays = origins.shape[0]
    rgb_chunks = []

    with torch.no_grad():
        for chunk_start in range(0, total_rays, args.chunk):
            chunk_end = min(chunk_start + args.chunk, total_rays)
            chunk_orig = origins[chunk_start:chunk_end]
            chunk_dir = dirs[chunk_start:chunk_end]
            # Need gradients for interface weighting, so temporarily enable autograd
            with torch.enable_grad():
                outputs = integrate_rays(
                    model,
                    chunk_orig,
                    chunk_dir,
                    num_samples=args.num_samples,
                    near=args.near,
                    far=args.far,
                    eps_rend=args.eps_rend,
                    kappa=args.kappa,
                    create_graph=False,
                )
            rgb_chunks.append(outputs["rgb"].detach())

    rgb = torch.cat(rgb_chunks, dim=0)
    rgb_image = rgb.view(dataset.height, dataset.width, 3).clamp(0.0, 1.0).cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, (rgb_image.numpy() * 255.0).astype("uint8"))
    print(f"Saved rendering to {output_path}")

    if args.compute_psnr and args.pose_split in {"train", "val", "test"}:
        gt = dataset.get_image(args.pose_index)
        mse = F.mse_loss(rgb_image, gt)
        psnr = -10.0 * torch.log10(mse + 1e-10)
        print(f"PSNR against ground truth: {psnr.item():.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render novel views from a trained phase-field NeRF")
    parser.add_argument("--dataset-root", default="data/nerf_synthetic/lego")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pose-split", default="val")
    parser.add_argument("--pose-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--eps-rend", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--compute-psnr", action="store_true")
    args = parser.parse_args()
    render_image(args)
