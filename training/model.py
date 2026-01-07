"""PointNet++ Semantic Segmentation Model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared Euclidean distance between each pair of points.

    Args:
        src: (B, N, C) source points
        dst: (B, M, C) target points

    Returns:
        dist: (B, N, M) squared distances
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest point sampling.

    Args:
        xyz: (B, N, 3) point coordinates
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint) sampled point indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index points from point cloud.

    Args:
        points: (B, N, C) input points
        idx: (B, S) sample indices

    Returns:
        new_points: (B, S, C) indexed points
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Query ball point grouping.

    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points

    Returns:
        group_idx: (B, S, nsample) grouped point indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction layer."""

    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: List[int], group_all: bool = False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) input coordinates
            points: (B, N, D) input features (or None)

        Returns:
            new_xyz: (B, npoint, 3) sampled coordinates
            new_points: (B, npoint, mlp[-1]) output features
        """
        if self.group_all:
            new_xyz, new_points = self.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self.sample_and_group(xyz, points)

        # new_points: (B, npoint, nsample, D+3)
        new_points = new_points.permute(0, 3, 2, 1)  # (B, D+3, nsample, npoint)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])

        return new_xyz, new_points

    def sample_and_group(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = xyz.shape
        S = self.npoint

        fps_idx = farthest_point_sample(xyz, S)
        new_xyz = index_points(xyz, fps_idx)

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        return new_xyz, new_points

    def sample_and_group_all(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = xyz.device
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C, device=device)
        grouped_xyz = xyz.view(B, 1, N, C)

        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """PointNet++ Feature Propagation layer."""

    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz1: (B, N, 3) input coordinates (higher resolution)
            xyz2: (B, S, 3) sampled coordinates (lower resolution)
            points1: (B, N, D1) input features (or None)
            points2: (B, S, D2) sampled features

        Returns:
            new_points: (B, N, mlp[-1]) propagated features
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # (B, D, N)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        return new_points.permute(0, 2, 1)  # (B, N, mlp[-1])


class PointNet2SemSeg(nn.Module):
    """PointNet++ for Semantic Segmentation."""

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # Set abstraction layers
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, in_channels + 3, [32, 32, 64])
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128])
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256])
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512])

        # Feature propagation layers
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + in_channels, [128, 128, 128])

        # Classifier
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, D) point features (optional)

        Returns:
            logits: (B, N, num_classes) per-point class logits
        """
        B, N, _ = xyz.shape

        # If no features, use xyz as features
        if features is None:
            l0_points = xyz
        else:
            l0_points = torch.cat([xyz, features], dim=-1)

        l0_xyz = xyz

        # Set abstraction
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature propagation
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # Classifier
        x = l0_points.permute(0, 2, 1)  # (B, 128, N)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # (B, N, num_classes)

        return x


if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    num_classes = 5
    model = PointNet2SemSeg(num_classes=num_classes, in_channels=3).to(device)

    # Test forward pass
    batch_size = 2
    num_points = 4096
    xyz = torch.randn(batch_size, num_points, 3).to(device)

    print(f"Input shape: {xyz.shape}")

    with torch.no_grad():
        output = model(xyz)

    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {num_points}, {num_classes})")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("Model test passed!")
