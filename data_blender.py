import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import imageio.v3 as iio
import numpy as np
import torch


def _load_image(path: str) -> torch.Tensor:
    img = iio.imread(path)
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return torch.from_numpy(img)


@dataclass
class FrameRecord:
    pose: torch.Tensor
    image: torch.Tensor


class BlenderDataset:
    """Minimal loader for Blender-formatted NeRF datasets."""

    def __init__(self, root: str, split: str = "train", frame_subset: Optional[Sequence[int]] = None) -> None:
        self.root = root
        self.split = split
        self.meta_path = os.path.join(root, f"transforms_{split}.json")
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Missing metadata file: {self.meta_path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.camera_angle_x = float(meta["camera_angle_x"])
        frames = meta["frames"]

        if frame_subset is not None:
            max_index = len(frames) - 1
            filtered: List[dict] = []
            for idx in frame_subset:
                if idx < 0 or idx > max_index:
                    raise IndexError(
                        f"Requested frame index {idx} is out of range for split '{split}' with {len(frames)} frames."
                    )
                filtered.append(frames[idx])
            frames = filtered
        self.frame_subset = frame_subset

        records: List[FrameRecord] = []
        for frame in frames:
            pose = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            fp = frame["file_path"]
            if fp.startswith("."):
                fp = fp[2:]
            image_path = os.path.join(root, fp)
            if not image_path.lower().endswith(".png"):
                image_path += ".png"
            image = _load_image(image_path)
            records.append(FrameRecord(pose=pose, image=image))

        self.records = records
        self.images = torch.stack([rec.image for rec in records], dim=0)  # [N, H, W, 3]
        self.poses = torch.stack([rec.pose for rec in records], dim=0)  # [N, 4, 4]
        self.num_images, self.height, self.width, _ = self.images.shape

        self.fx = 0.5 * self.width / math.tan(0.5 * self.camera_angle_x)
        self.fy = self.fx

        ii, jj = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            indexing="ij",
        )
        self.pixel_i = ii
        self.pixel_j = jj

    def sample_random_rays(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idxs = torch.randint(0, self.num_images, (batch_size,), dtype=torch.long)
        rows = torch.randint(0, self.height, (batch_size,), dtype=torch.long)
        cols = torch.randint(0, self.width, (batch_size,), dtype=torch.long)

        target_rgb = self.images[idxs, rows, cols]
        poses = self.poses[idxs]

        rows_f = rows.to(torch.float32)
        cols_f = cols.to(torch.float32)
        dirs_cam = torch.stack([
            (cols_f + 0.5 - 0.5 * self.width) / self.fx,
            -(rows_f + 0.5 - 0.5 * self.height) / self.fy,
            -torch.ones_like(cols_f),
        ], dim=-1)

        rot = poses[:, :3, :3]
        trans = poses[:, :3, 3]
        dirs_world = torch.bmm(rot, dirs_cam.unsqueeze(-1)).squeeze(-1)
        dirs_world = dirs_world / torch.linalg.norm(dirs_world, dim=-1, keepdim=True)

        origins = trans
        return origins.to(device), dirs_world.to(device), target_rgb.to(device)

    def get_pose(self, index: int) -> torch.Tensor:
        return self.poses[index]

    def get_image(self, index: int) -> torch.Tensor:
        return self.images[index]

    def __len__(self) -> int:
        return self.num_images

    def generate_rays_for_pose(self, pose: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        ii = self.pixel_i.reshape(-1).to(device)
        jj = self.pixel_j.reshape(-1).to(device)
        dirs_cam = torch.stack([
            (jj + 0.5 - 0.5 * self.width) / self.fx,
            -(ii + 0.5 - 0.5 * self.height) / self.fy,
            -torch.ones_like(jj),
        ], dim=-1)
        rot = pose[:3, :3].to(device)
        trans = pose[:3, 3].to(device)
        dirs_world = torch.matmul(dirs_cam, rot.T)
        dirs_world = dirs_world / torch.linalg.norm(dirs_world, dim=-1, keepdim=True)
        origins = trans.expand_as(dirs_world)
        return origins, dirs_world
