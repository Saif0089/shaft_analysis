import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
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
z_min, z_max = points[:, 2].min(), points[:, 2].max()

# Radial distance
radial_dist = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)

# Initialize labels
labels = np.full(len(points), -1, dtype=np.int32)

print("\n" + "="*60)
print("FINAL SEGMENTATION PIPELINE")
print("="*60)

# ============================================================
# STEP 0: Compute normals
# ============================================================
print("\n[0] Computing normals...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=25))
normals = np.asarray(pcd.normals)
normal_z = np.abs(normals[:, 2])

# ============================================================
# SEGMENT 1: OUTER WALL
# ============================================================
print("\n[1] OUTER WALL...")
outer_wall_mask = radial_dist > (outer_radius - wall_thickness)
labels[outer_wall_mask] = 1
print(f"    {np.sum(outer_wall_mask):,} points ({100*np.sum(outer_wall_mask)/len(points):.1f}%)")

# ============================================================
# SEGMENT 2: Find platform Z-levels and create platform zones
# ============================================================
print("\n[2] PLATFORM ZONES (walkways at specific Z-levels)...")

# Use fine Z-histogram to find density peaks
z_hist, z_edges = np.histogram(points[labels == -1, 2], bins=400)
z_centers = (z_edges[:-1] + z_edges[1:]) / 2

# Find peaks (platforms have high point density)
peaks, props = find_peaks(z_hist, height=np.percentile(z_hist, 80), distance=8, prominence=200)
platform_z_levels = sorted(z_centers[peaks])

# Merge close platforms
merged = []
for z in platform_z_levels:
    if not merged or z - merged[-1] > 0.4:
        merged.append(z)
    else:
        merged[-1] = (merged[-1] + z) / 2
platform_z_levels = merged

print(f"    Found {len(platform_z_levels)} platform levels")

# Platform zones: Z ± 0.25m around each platform level
platform_zone_half = 0.25
total_platform = 0

for z_level in platform_z_levels:
    # Points in this platform zone (excluding outer wall)
    zone_mask = (
        (np.abs(points[:, 2] - z_level) < platform_zone_half) &
        (labels == -1)
    )

    if np.sum(zone_mask) > 500:
        labels[zone_mask] = 2
        total_platform += np.sum(zone_mask)
        print(f"      Z={z_level:.3f}m: {np.sum(zone_mask):,} points")

print(f"    Total platform zone: {total_platform:,} points ({100*total_platform/len(points):.1f}%)")

# ============================================================
# SEGMENT 3: CENTRAL COLUMN
# ============================================================
print("\n[3] CENTRAL COLUMN...")
central_mask = (radial_dist < 1.3) & (labels == -1)
labels[central_mask] = 3
print(f"    {np.sum(central_mask):,} points ({100*np.sum(central_mask)/len(points):.1f}%)")

# ============================================================
# SEGMENT 4: HORIZONTAL SURFACES (remaining)
# ============================================================
print("\n[4] HORIZONTAL SURFACES (remaining)...")
horiz_mask = (normal_z > 0.65) & (labels == -1)
labels[horiz_mask] = 4
print(f"    {np.sum(horiz_mask):,} points ({100*np.sum(horiz_mask)/len(points):.1f}%)")

# ============================================================
# SEGMENT 5: VERTICAL STRUCTURES (supports, edges)
# ============================================================
print("\n[5] VERTICAL STRUCTURES...")
vert_mask = (normal_z < 0.35) & (labels == -1)
labels[vert_mask] = 5
print(f"    {np.sum(vert_mask):,} points ({100*np.sum(vert_mask)/len(points):.1f}%)")

# ============================================================
# SEGMENT 6: DIAGONAL/ANGLED STRUCTURES
# ============================================================
print("\n[6] DIAGONAL STRUCTURES...")
remaining = labels == -1
labels[remaining] = 6
print(f"    {np.sum(remaining):,} points ({100*np.sum(remaining)/len(points):.1f}%)")

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
    4: "Horizontal Surfaces",
    5: "Vertical Structures",
    6: "Diagonal/Other"
}

for seg_id, name in segment_names.items():
    count = np.sum(labels == seg_id)
    pct = 100*count/len(points)
    print(f"  {seg_id}. {name}: {count:,} points ({pct:.1f}%)")

# ============================================================
# VISUALIZATION
# ============================================================
print("\nGenerating visualizations...")

colors_map = {
    1: [0.2, 0.4, 0.8],    # Blue - outer wall
    2: [1.0, 0.6, 0.0],    # Orange - platforms
    3: [0.2, 0.7, 0.2],    # Green - central
    4: [0.9, 0.2, 0.2],    # Red - horizontal
    5: [0.6, 0.3, 0.7],    # Purple - vertical
    6: [0.5, 0.5, 0.5],    # Gray - diagonal
}

# Figure 1: Main overview
fig1, axes = plt.subplots(2, 3, figsize=(18, 12))

# Top view
ax = axes[0, 0]
for seg_id, name in segment_names.items():
    mask = labels == seg_id
    if np.sum(mask) > 0:
        sample = np.random.choice(np.where(mask)[0], min(25000, np.sum(mask)), replace=False)
        ax.scatter(points[sample, 0], points[sample, 1], s=0.3, alpha=0.6,
                   c=[colors_map[seg_id]], label=name)
ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Top View (XY)')
ax.legend(markerscale=10, fontsize=8)

# Front view XZ
ax = axes[0, 1]
for seg_id in segment_names.keys():
    mask = labels == seg_id
    if np.sum(mask) > 0:
        sample = np.random.choice(np.where(mask)[0], min(25000, np.sum(mask)), replace=False)
        ax.scatter(points[sample, 0], points[sample, 2], s=0.3, alpha=0.5, c=[colors_map[seg_id]])
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Front View (XZ)')

# Side view YZ
ax = axes[0, 2]
for seg_id in segment_names.keys():
    mask = labels == seg_id
    if np.sum(mask) > 0:
        sample = np.random.choice(np.where(mask)[0], min(25000, np.sum(mask)), replace=False)
        ax.scatter(points[sample, 1], points[sample, 2], s=0.3, alpha=0.5, c=[colors_map[seg_id]])
ax.set_xlabel('Y (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Side View (YZ)')

# Key segments
for i, seg_id in enumerate([2, 5, 4]):
    ax = axes[1, i]
    mask = labels == seg_id
    if np.sum(mask) > 100:
        sample = np.random.choice(np.where(mask)[0], min(30000, np.sum(mask)), replace=False)
        ax.scatter(points[sample, 0], points[sample, 1], s=0.5, alpha=0.7, c=[colors_map[seg_id]])
    circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{segment_names[seg_id]} ({np.sum(mask):,})')
    ax.set_xlim(-15, -6)
    ax.set_ylim(-5, 4)

plt.tight_layout()
plt.savefig('visualizations/final_overview.png', dpi=150)
print("Saved: visualizations/final_overview.png")

# Figure 2: Z-slices
fig2 = plt.figure(figsize=(20, 10))
z_slices = np.linspace(z_min + 0.5, z_max - 0.5, 8)

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
    ax.set_title(f'Z = {z:.2f}m ({np.sum(slice_mask):,})')
    ax.set_xlim(-15, -6)
    ax.set_ylim(-5, 4)
    if i == 0:
        ax.legend(markerscale=5, fontsize=6, loc='upper right')

plt.tight_layout()
plt.savefig('visualizations/final_slices.png', dpi=150)
print("Saved: visualizations/final_slices.png")

# Figure 3: Side views for all segments
fig3 = plt.figure(figsize=(18, 10))
for i, (seg_id, name) in enumerate(segment_names.items()):
    ax = fig3.add_subplot(2, 3, i + 1)
    mask = labels == seg_id
    if np.sum(mask) > 50:
        sample = np.random.choice(np.where(mask)[0], min(35000, np.sum(mask)), replace=False)
        ax.scatter(points[sample, 0], points[sample, 2], s=0.4, alpha=0.6, c=[colors_map[seg_id]])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'{name} ({np.sum(mask):,})')

plt.tight_layout()
plt.savefig('visualizations/final_side_views.png', dpi=150)
print("Saved: visualizations/final_side_views.png")

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

output_las.write('segmented_shaft_final.las')
print("Saved: segmented_shaft_final.las")

# Save individual segments
import os
os.makedirs('segments_final', exist_ok=True)
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
        seg_las.write(f'segments_final/{seg_id}_{safe_name}.las')
        print(f"  {seg_id}_{safe_name}.las: {np.sum(mask):,} points")

# Save summary report
print("\nSaving summary report...")
with open('segmentation_report.txt', 'w') as f:
    f.write("SHAFT POINT CLOUD SEGMENTATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Input file: ../Navvis5mmShaft14_middle_10m.las\n")
    f.write(f"Total points: {len(points):,}\n\n")
    f.write("STRUCTURE MEASUREMENTS:\n")
    f.write(f"  Shaft diameter: {2*outer_radius:.2f}m\n")
    f.write(f"  Shaft height: {z_max - z_min:.2f}m\n")
    f.write(f"  Center: ({center_x:.3f}, {center_y:.3f})\n")
    f.write(f"  Platform levels: {len(platform_z_levels)}\n")
    for i, z in enumerate(platform_z_levels):
        f.write(f"    Level {i+1}: Z = {z:.3f}m\n")
    f.write("\nSEGMENTATION RESULTS:\n")
    for seg_id, name in segment_names.items():
        count = np.sum(labels == seg_id)
        f.write(f"  {seg_id}. {name}: {count:,} points ({100*count/len(points):.1f}%)\n")
    f.write("\nOUTPUT FILES:\n")
    f.write("  segmented_shaft_final.las - Full segmented point cloud\n")
    f.write("  segments_final/ - Individual segment files\n")
    f.write("  visualizations/ - Visualization images\n")

print("\nSegmentation complete!")
print(f"\nOutput files in: {os.getcwd()}")
