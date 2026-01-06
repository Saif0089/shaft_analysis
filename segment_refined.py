import laspy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Load the LAS point cloud
print("Loading point cloud...")
las = laspy.read("../Navvis5mmShaft14_middle_10m.las")
points = np.vstack([las.x, las.y, las.z]).T
intensity = las.intensity
print(f"Loaded {len(points):,} points")

# Measurements
center_x, center_y = -10.5549, -0.5347
outer_radius = 3.79
wall_thickness = 0.18

# Radial distance
radial_dist = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)

# Initialize labels
labels = np.full(len(points), -1, dtype=np.int32)

print("\n" + "="*60)
print("REFINED SEGMENTATION")
print("="*60)

# Compute normals
print("\n[0] Computing normals...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=25))
normals = np.asarray(pcd.normals)
normal_z = np.abs(normals[:, 2])

# ============================================================
# 1. OUTER WALL
# ============================================================
print("\n[1] OUTER WALL...")
outer_wall_mask = radial_dist > (outer_radius - wall_thickness)
labels[outer_wall_mask] = 1
print(f"    {np.sum(outer_wall_mask):,} points")

# ============================================================
# 2. PLATFORM ZONES (walkways at specific Z-levels)
# ============================================================
print("\n[2] PLATFORM ZONES...")
from scipy.signal import find_peaks
z_hist, z_edges = np.histogram(points[labels == -1, 2], bins=400)
z_centers = (z_edges[:-1] + z_edges[1:]) / 2
peaks, _ = find_peaks(z_hist, height=np.percentile(z_hist, 80), distance=8, prominence=200)
platform_z_levels = sorted(z_centers[peaks])

# Merge close platforms
merged = []
for z in platform_z_levels:
    if not merged or z - merged[-1] > 0.4:
        merged.append(z)
platform_z_levels = merged

platform_zone_half = 0.25
for z_level in platform_z_levels:
    zone_mask = (np.abs(points[:, 2] - z_level) < platform_zone_half) & (labels == -1)
    labels[zone_mask] = 2
print(f"    {np.sum(labels == 2):,} points at {len(platform_z_levels)} levels")

# ============================================================
# 3. CENTRAL COLUMN
# ============================================================
print("\n[3] CENTRAL COLUMN...")
central_mask = (radial_dist < 1.3) & (labels == -1)
labels[central_mask] = 3
print(f"    {np.sum(central_mask):,} points")

# ============================================================
# 4. BOTTOM STRUCTURE (Z < -524.5m, the mystery platform/guard)
# ============================================================
print("\n[4] BOTTOM STRUCTURE (mystery platform)...")
bottom_z_threshold = -524.5
bottom_mask = (points[:, 2] < bottom_z_threshold) & (labels == -1)
labels[bottom_mask] = 4
print(f"    {np.sum(bottom_mask):,} points (Z < {bottom_z_threshold}m)")

# ============================================================
# 5. PIPES/WIRES (vertical surfaces near wall, r > 2.8m)
# ============================================================
print("\n[5] PIPES/WIRES (near wall)...")
pipes_radius_threshold = 2.8
pipes_mask = (
    (normal_z < 0.5) &  # Vertical surface
    (radial_dist > pipes_radius_threshold) &
    (labels == -1)
)
labels[pipes_mask] = 5
print(f"    {np.sum(pipes_mask):,} points (r > {pipes_radius_threshold}m)")

# ============================================================
# 6. VERTICAL STEEL GUARDS (vertical surfaces, more central, r < 2.8m)
# ============================================================
print("\n[6] VERTICAL STEEL GUARDS (central)...")
guards_mask = (
    (normal_z < 0.5) &  # Vertical surface
    (radial_dist <= pipes_radius_threshold) &
    (labels == -1)
)
labels[guards_mask] = 6
print(f"    {np.sum(guards_mask):,} points (r <= {pipes_radius_threshold}m)")

# ============================================================
# 7. HORIZONTAL SURFACES (remaining)
# ============================================================
print("\n[7] HORIZONTAL SURFACES...")
horiz_mask = (normal_z > 0.5) & (labels == -1)
labels[horiz_mask] = 7
print(f"    {np.sum(horiz_mask):,} points")

# ============================================================
# 8. OTHER
# ============================================================
remaining = np.sum(labels == -1)
labels[labels == -1] = 8
print(f"\n[8] OTHER: {remaining:,} points")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("SEGMENTATION SUMMARY")
print("="*60)
segment_names = {
    1: "Outer Wall",
    2: "Platform Zones",
    3: "Central Column",
    4: "Bottom Structure",
    5: "Pipes/Wires",
    6: "Vertical Guards",
    7: "Horizontal Surfaces",
    8: "Other"
}

for seg_id, name in segment_names.items():
    count = np.sum(labels == seg_id)
    print(f"  {seg_id}. {name}: {count:,} points ({100*count/len(points):.1f}%)")

# ============================================================
# VISUALIZATION
# ============================================================
print("\nGenerating visualizations...")

colors_map = {
    1: [0.2, 0.4, 0.8],    # Blue - outer wall
    2: [1.0, 0.6, 0.0],    # Orange - platforms
    3: [0.2, 0.7, 0.2],    # Green - central column
    4: [0.5, 0.0, 0.5],    # Dark purple - bottom structure
    5: [0.0, 0.8, 0.8],    # Cyan - pipes/wires
    6: [0.9, 0.2, 0.2],    # Red - vertical guards
    7: [0.9, 0.9, 0.2],    # Yellow - horizontal
    8: [0.5, 0.5, 0.5],    # Gray - other
}

# Figure 1: Main overview
fig1, axes = plt.subplots(2, 4, figsize=(22, 11))

# Top view - all
ax = axes[0, 0]
for seg_id, name in segment_names.items():
    mask = labels == seg_id
    if np.sum(mask) > 0:
        sample = np.random.choice(np.where(mask)[0], min(20000, np.sum(mask)), replace=False)
        ax.scatter(points[sample, 0], points[sample, 1], s=0.3, alpha=0.6,
                   c=[colors_map[seg_id]], label=name)
ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Top View - All Segments')
ax.legend(markerscale=10, fontsize=7, loc='upper right')

# Side view XZ - all
ax = axes[0, 1]
for seg_id in segment_names.keys():
    mask = labels == seg_id
    if np.sum(mask) > 0:
        sample = np.random.choice(np.where(mask)[0], min(20000, np.sum(mask)), replace=False)
        ax.scatter(points[sample, 0], points[sample, 2], s=0.3, alpha=0.5, c=[colors_map[seg_id]])
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Side View (XZ)')

# Pipes/Wires only
ax = axes[0, 2]
mask = labels == 5
if np.sum(mask) > 100:
    sample = np.random.choice(np.where(mask)[0], min(30000, np.sum(mask)), replace=False)
    ax.scatter(points[sample, 0], points[sample, 1], s=0.5, alpha=0.7, c=[colors_map[5]])
circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='gray', linestyle='--')
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_title(f'Pipes/Wires ({np.sum(mask):,})')
ax.set_xlim(-15, -6)
ax.set_ylim(-5, 4)

# Vertical Guards only
ax = axes[0, 3]
mask = labels == 6
if np.sum(mask) > 100:
    sample = np.random.choice(np.where(mask)[0], min(30000, np.sum(mask)), replace=False)
    ax.scatter(points[sample, 0], points[sample, 1], s=0.5, alpha=0.7, c=[colors_map[6]])
circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='gray', linestyle='--')
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_title(f'Vertical Guards ({np.sum(mask):,})')
ax.set_xlim(-15, -6)
ax.set_ylim(-5, 4)

# Bottom structure
ax = axes[1, 0]
mask = labels == 4
if np.sum(mask) > 100:
    sample = np.random.choice(np.where(mask)[0], min(30000, np.sum(mask)), replace=False)
    ax.scatter(points[sample, 0], points[sample, 1], s=0.5, alpha=0.7, c=[colors_map[4]])
circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='gray', linestyle='--')
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_title(f'Bottom Structure ({np.sum(mask):,})')
ax.set_xlim(-15, -6)
ax.set_ylim(-5, 4)

# Platform zones
ax = axes[1, 1]
mask = labels == 2
if np.sum(mask) > 100:
    sample = np.random.choice(np.where(mask)[0], min(30000, np.sum(mask)), replace=False)
    ax.scatter(points[sample, 0], points[sample, 1], s=0.5, alpha=0.7, c=[colors_map[2]])
circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='gray', linestyle='--')
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_title(f'Platform Zones ({np.sum(mask):,})')
ax.set_xlim(-15, -6)
ax.set_ylim(-5, 4)

# Side view - Pipes vs Guards
ax = axes[1, 2]
for seg_id in [5, 6]:
    mask = labels == seg_id
    if np.sum(mask) > 100:
        sample = np.random.choice(np.where(mask)[0], min(20000, np.sum(mask)), replace=False)
        ax.scatter(points[sample, 0], points[sample, 2], s=0.5, alpha=0.5,
                   c=[colors_map[seg_id]], label=segment_names[seg_id])
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Pipes vs Guards - Side View')
ax.legend(markerscale=10)

# Side view - Bottom structure
ax = axes[1, 3]
mask = labels == 4
if np.sum(mask) > 100:
    sample = np.random.choice(np.where(mask)[0], min(30000, np.sum(mask)), replace=False)
    ax.scatter(points[sample, 0], points[sample, 2], s=0.5, alpha=0.7, c=[colors_map[4]])
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_title(f'Bottom Structure - Side View')

plt.tight_layout()
plt.savefig('visualizations/refined_overview.png', dpi=150)
print("Saved: visualizations/refined_overview.png")

# Figure 2: Z-slices
fig2 = plt.figure(figsize=(20, 10))
z_slices = np.linspace(points[:, 2].min() + 0.5, points[:, 2].max() - 0.5, 8)

for i, z in enumerate(z_slices):
    ax = fig2.add_subplot(2, 4, i + 1)
    slice_mask = np.abs(points[:, 2] - z) < 0.3
    slice_pts = points[slice_mask]
    slice_labels = labels[slice_mask]

    for seg_id in segment_names.keys():
        seg_mask = slice_labels == seg_id
        if np.sum(seg_mask) > 0:
            ax.scatter(slice_pts[seg_mask, 0], slice_pts[seg_mask, 1],
                      s=1.5, alpha=0.7, c=[colors_map[seg_id]], label=segment_names[seg_id])

    circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title(f'Z = {z:.2f}m')
    ax.set_xlim(-15, -6)
    ax.set_ylim(-5, 4)
    if i == 0:
        ax.legend(markerscale=4, fontsize=6, loc='upper right')

plt.tight_layout()
plt.savefig('visualizations/refined_slices.png', dpi=150)
print("Saved: visualizations/refined_slices.png")

# Save segmented point cloud
print("\nSaving segmented point cloud...")
output_las = laspy.create(point_format=las.point_format, file_version=las.header.version)
output_las.x = las.x
output_las.y = las.y
output_las.z = las.z
output_las.intensity = las.intensity
output_las.classification = labels.astype(np.uint8)

red = np.zeros(len(points), dtype=np.uint16)
green = np.zeros(len(points), dtype=np.uint16)
blue = np.zeros(len(points), dtype=np.uint16)

for seg_id, color in colors_map.items():
    mask = labels == seg_id
    red[mask] = int(color[0] * 65535)
    green[mask] = int(color[1] * 65535)
    blue[mask] = int(color[2] * 65535)

if hasattr(output_las, 'red'):
    output_las.red = red
    output_las.green = green
    output_las.blue = blue

output_las.write('segmented_shaft_refined.las')
print("Saved: segmented_shaft_refined.las")

# Save individual segments
import os
os.makedirs('segments_refined', exist_ok=True)
print("\nSaving individual segment files...")
for seg_id, name in segment_names.items():
    mask = labels == seg_id
    if np.sum(mask) > 0:
        seg_las = laspy.create(point_format=las.point_format, file_version=las.header.version)
        seg_las.x = las.x[mask]
        seg_las.y = las.y[mask]
        seg_las.z = las.z[mask]
        seg_las.intensity = las.intensity[mask]
        safe_name = name.lower().replace("/", "_").replace(" ", "_")
        seg_las.write(f'segments_refined/{seg_id}_{safe_name}.las')
        print(f"  {seg_id}_{safe_name}.las: {np.sum(mask):,} points")

# Update report
with open('segmentation_report_refined.txt', 'w') as f:
    f.write("REFINED SHAFT SEGMENTATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write("SEGMENTATION RESULTS:\n")
    for seg_id, name in segment_names.items():
        count = np.sum(labels == seg_id)
        f.write(f"  {seg_id}. {name}: {count:,} points ({100*count/len(points):.1f}%)\n")
    f.write(f"\nKey parameters:\n")
    f.write(f"  - Pipes/Wires: r > {pipes_radius_threshold}m (near wall)\n")
    f.write(f"  - Vertical Guards: r <= {pipes_radius_threshold}m (central)\n")
    f.write(f"  - Bottom Structure: Z < {bottom_z_threshold}m\n")

print("\nRefined segmentation complete!")
