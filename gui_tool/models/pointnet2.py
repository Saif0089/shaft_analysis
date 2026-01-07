"""
PointNet++ for Point Cloud Segmentation
Implements hierarchical point set learning with set abstraction and feature propagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared Euclidean distance between two point sets.
    src: (B, N, C)
    dst: (B, M, C)
    Returns: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest point sampling to select npoint points from xyz.
    xyz: (B, N, 3)
    Returns: (B, npoint) indices
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
    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    Returns: (B, S, C) or (B, S, K, C)
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
    Ball query - find all points within radius.
    xyz: (B, N, 3) all points
    new_xyz: (B, S, 3) query points
    Returns: (B, S, nsample) indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # Handle case where fewer than nsample points in ball
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor,
                     points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample and group operation for set abstraction.
    xyz: (B, N, 3)
    points: (B, N, D) or None
    Returns: new_xyz (B, npoint, 3), new_points (B, npoint, nsample, 3+D)
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample all points as one group (for final layer).
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)

    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """Set Abstraction Layer - downsamples and extracts features"""

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: List[int], group_all: bool = False):
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
        xyz: (B, N, 3)
        points: (B, N, D)
        Returns: new_xyz (B, npoint, 3), new_points (B, npoint, mlp[-1])
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # (B, npoint, nsample, 3+D) -> (B, 3+D, nsample, npoint)
        new_points = new_points.permute(0, 3, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Max pooling over samples
        new_points = torch.max(new_points, 2)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """Feature Propagation Layer - upsamples features"""

    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        """
        xyz1: (B, N, 3) - target points (more points)
        xyz2: (B, S, 3) - source points (fewer points)
        points1: (B, N, D1) - target features (skip connection)
        points2: (B, S, D2) - source features to upsample
        Returns: (B, N, mlp[-1])
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3 nearest neighbors

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # (B, D, N)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points.permute(0, 2, 1)  # (B, N, mlp[-1])


class PointNet2Segmentation(nn.Module):
    """PointNet++ for semantic segmentation"""

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # Set Abstraction layers (encoder)
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32,
            in_channel=in_channels + 3, mlp=[32, 32, 64]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32,
            in_channel=64 + 3, mlp=[64, 64, 128]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=0.8, nsample=32,
            in_channel=256 + 3, mlp=[256, 256, 512]
        )

        # Feature Propagation layers (decoder)
        self.fp4 = PointNetFeaturePropagation(in_channel=768, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=320, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + in_channels, mlp=[128, 128, 128])

        # Segmentation head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        xyz: (B, N, 3) point coordinates
        features: (B, N, C) additional point features (colors, normals, etc.)
        Returns: (B, N, num_classes) per-point class logits
        """
        B, N, _ = xyz.shape

        if features is not None:
            l0_points = features
        else:
            l0_points = None
        l0_xyz = xyz

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Decoder with skip connections
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # Segmentation head
        x = l0_points.permute(0, 2, 1)  # (B, 128, N)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)  # (B, num_classes, N)
        x = x.permute(0, 2, 1)  # (B, N, num_classes)

        return x


class PointNet2SegmentationMSG(nn.Module):
    """PointNet++ with Multi-Scale Grouping for better accuracy"""

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # SA1 with multi-scale grouping
        self.sa1 = PointNetSetAbstractionMSG(
            npoint=1024,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=in_channels,
            mlp_list=[[16, 16, 32], [32, 32, 64]]
        )
        self.sa2 = PointNetSetAbstractionMSG(
            npoint=256,
            radius_list=[0.1, 0.2],
            nsample_list=[16, 32],
            in_channel=32 + 64,
            mlp_list=[[64, 64, 128], [64, 96, 128]]
        )
        self.sa3 = PointNetSetAbstractionMSG(
            npoint=64,
            radius_list=[0.2, 0.4],
            nsample_list=[16, 32],
            in_channel=128 + 128,
            mlp_list=[[128, 196, 256], [128, 196, 256]]
        )
        self.sa4 = PointNetSetAbstractionMSG(
            npoint=16,
            radius_list=[0.4, 0.8],
            nsample_list=[16, 32],
            in_channel=256 + 256,
            mlp_list=[[256, 256, 512], [256, 384, 512]]
        )

        # Feature Propagation
        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 512 + 256 + 256, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=128 + 128 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=32 + 64 + 256, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + in_channels, mlp=[128, 128, 128])

        # Segmentation head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = xyz.shape

        l0_points = features
        l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = l0_points.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)

        return x


class PointNetSetAbstractionMSG(nn.Module):
    """Set Abstraction with Multi-Scale Grouping"""

    def __init__(self, npoint: int, radius_list: List[float], nsample_list: List[int],
                 in_channel: int, mlp_list: List[List[int]]):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3

            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel

            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = xyz.shape
        S = self.npoint

        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))

        new_points_list = []
        for i, (radius, nsample) in enumerate(zip(self.radius_list, self.nsample_list)):
            idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)

            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)

            for j, conv in enumerate(self.conv_blocks[i]):
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))

            new_points = torch.max(grouped_points, 2)[0]
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1)
        new_points_concat = new_points_concat.permute(0, 2, 1)

        return new_xyz, new_points_concat


class PointNet2SegmentationLight(nn.Module):
    """Lightweight PointNet++ for small datasets - fewer parameters, less overfitting"""

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # Smaller Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=16,
            in_channel=in_channels + 3, mlp=[32, 32, 64]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=16,
            in_channel=64 + 3, mlp=[64, 64, 128]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=32, radius=0.8, nsample=16,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )

        # Feature Propagation layers
        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp2 = PointNetFeaturePropagation(in_channel=192, mlp=[128, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + in_channels, mlp=[128, 64, 64])

        # Segmentation head with dropout
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(64, num_classes, 1)

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = xyz.shape

        l0_points = features
        l0_xyz = xyz

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # Segmentation head
        x = l0_points.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)

        return x


if __name__ == '__main__':
    # Test the models
    print("Testing PointNet2Segmentation:")
    model = PointNet2Segmentation(num_classes=8, in_channels=3)
    xyz = torch.randn(2, 4096, 3)
    features = torch.randn(2, 4096, 3)

    output = model(xyz, features)
    print(f"  Input: xyz {xyz.shape}, features {features.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting PointNet2SegmentationLight:")
    model_light = PointNet2SegmentationLight(num_classes=8, in_channels=3)
    output = model_light(xyz, features)
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_light.parameters()):,}")
