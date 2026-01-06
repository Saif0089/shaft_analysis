import laspy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
import open3d as o3d

# Load the LAS point cloud
print("Loading point cloud...")
las = laspy.read("../Navvis5mmShaft14_middle_10m.las")
points = np.vstack([las.x, las.y, las.z]).T
intensity = las.intensity
print(f"Loaded {len(points):,} points")

# Measurements from previous analysis
center_x, center_y = -10.5549, -0.5347
outer_radius = 3.79
z_min, z_max = points[:, 2].min(), points[:, 2].max()

# Calculate radial distance for each point
radial_dist = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)

# Initialize segment labels (-1 = unassigned)
labels = np.full(len(points), -1, dtype=np.int32)

print("\n" + "="*60)
print("SEGMENTATION PIPELINE")
print("="*60)

# ============================================================
# SEGMENT 1: OUTER WALL (cylindrical shell)
# Points near the outer radius
# ============================================================
print("\n[1] Segmenting OUTER WALL...")
wall_thickness = 0.15  # 15cm wall thickness tolerance
outer_wall_mask = (radial_dist > outer_radius - wall_thickness) & (radial_dist < outer_radius + 0.05)
labels[outer_wall_mask] = 1
print(f"    Outer wall points: {np.sum(outer_wall_mask):,} ({100*np.sum(outer_wall_mask)/len(points):.1f}%)")

# ============================================================
# SEGMENT 2: FLOORS/PLATFORMS (horizontal surfaces)
# Use Z-histogram to find floor levels, then segment horizontal regions
# ============================================================
print("\n[2] Segmenting FLOORS/PLATFORMS...")

# Fine-grained Z analysis
z_hist, z_edges = np.histogram(points[:, 2], bins=1000)
z_centers = (z_edges[:-1] + z_edges[1:]) / 2

# Find significant peaks (floors have high point density)
peaks, properties = find_peaks(z_hist, height=np.percentile(z_hist, 90), distance=5, prominence=200)
floor_z_levels = z_centers[peaks]

# Filter to get distinct floors (merge close ones)
merged_floors = []
for z in sorted(floor_z_levels):
    if not merged_floors or z - merged_floors[-1] > 0.3:  # 30cm minimum between floors
        merged_floors.append(z)
    else:
        # Merge with previous by averaging
        merged_floors[-1] = (merged_floors[-1] + z) / 2

floor_z_levels = np.array(merged_floors)
print(f"    Detected {len(floor_z_levels)} floor levels:")

floor_thickness = 0.12  # 12cm thickness for floor detection
floor_points_count = 0
for i, z in enumerate(floor_z_levels):
    floor_mask = (np.abs(points[:, 2] - z) < floor_thickness) & (labels == -1)
    # Only include interior points (not outer wall)
    floor_mask &= (radial_dist < outer_radius - wall_thickness)
    labels[floor_mask] = 2
    count = np.sum(floor_mask)
    floor_points_count += count
    print(f"      Floor at Z={z:.3f}m: {count:,} points")

print(f"    Total floor points: {floor_points_count:,} ({100*floor_points_count/len(points):.1f}%)")

# ============================================================
# SEGMENT 3: CENTRAL COLUMN/CORE STRUCTURE
# Points near the center of the shaft
# ============================================================
print("\n[3] Segmenting CENTRAL STRUCTURE...")
# From radial analysis, there's structure around r=1.6m
inner_radius_threshold = 1.8  # Points closer than this to center
central_mask = (radial_dist < inner_radius_threshold) & (labels == -1)
labels[central_mask] = 3
print(f"    Central structure points: {np.sum(central_mask):,} ({100*np.sum(central_mask)/len(points):.1f}%)")

# ============================================================
# SEGMENT 4: VERTICAL STRUCTURES (columns, supports)
# Use normal estimation and vertical orientation filtering
# ============================================================
print("\n[4] Segmenting VERTICAL STRUCTURES...")

# Work with remaining unassigned points
remaining_mask = labels == -1
remaining_indices = np.where(remaining_mask)[0]
remaining_points = points[remaining_mask]

if len(remaining_points) > 1000:
    # Create Open3D point cloud for normal estimation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(remaining_points)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    normals = np.asarray(pcd.normals)

    # Vertical structures have horizontal normals (normal perpendicular to Z)
    # Normal Z component should be close to 0
    normal_z = np.abs(normals[:, 2])
    vertical_surface_mask = normal_z < 0.3  # Normal mostly horizontal = vertical surface

    # Apply to global labels
    vertical_indices = remaining_indices[vertical_surface_mask]
    labels[vertical_indices] = 4
    print(f"    Vertical structure points: {len(vertical_indices):,} ({100*len(vertical_indices)/len(points):.1f}%)")

# ============================================================
# SEGMENT 5: RAILINGS/GUARDS (thin horizontal structures above floors)
# Points just above floor levels that aren't floor
# ============================================================
print("\n[5] Segmenting RAILINGS/GUARDS...")
railing_height_min = 0.3  # 30cm above floor
railing_height_max = 1.2  # up to 1.2m above floor
railing_count = 0

for floor_z in floor_z_levels:
    railing_mask = (
        (points[:, 2] > floor_z + railing_height_min) &
        (points[:, 2] < floor_z + railing_height_max) &
        (labels == -1) &
        (radial_dist < outer_radius - wall_thickness)  # Interior only
    )
    labels[railing_mask] = 5
    railing_count += np.sum(railing_mask)

print(f"    Railing points: {railing_count:,} ({100*railing_count/len(points):.1f}%)")

# ============================================================
# SEGMENT 6: REMAINING UNASSIGNED -> OTHER STRUCTURES
# ============================================================
remaining = np.sum(labels == -1)
labels[labels == -1] = 6
print(f"\n[6] Other/unassigned points: {remaining:,} ({100*remaining/len(points):.1f}%)")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("SEGMENTATION SUMMARY")
print("="*60)
segment_names = {
    1: "Outer Wall",
    2: "Floors/Platforms",
    3: "Central Structure",
    4: "Vertical Structures",
    5: "Railings/Guards",
    6: "Other"
}

for seg_id, name in segment_names.items():
    count = np.sum(labels == seg_id)
    print(f"  {seg_id}. {name}: {count:,} points ({100*count/len(points):.1f}%)")

# ============================================================
# VISUALIZATION
# ============================================================
print("\nGenerating visualizations...")

# Color map for segments
colors = plt.cm.tab10(np.linspace(0, 1, 10))
segment_colors = {
    1: colors[0],  # Outer wall - blue
    2: colors[1],  # Floors - orange
    3: colors[2],  # Central - green
    4: colors[3],  # Vertical - red
    5: colors[4],  # Railings - purple
    6: colors[7],  # Other - gray
}

# Create large figure with multiple views
fig = plt.figure(figsize=(24, 20))

# 1. XY view (top-down) colored by segment
ax1 = fig.add_subplot(3, 4, 1)
for seg_id in segment_names.keys():
    mask = labels == seg_id
    if np.sum(mask) > 0:
        sample = np.random.choice(np.where(mask)[0], min(10000, np.sum(mask)), replace=False)
        ax1.scatter(points[sample, 0], points[sample, 1], s=0.5, alpha=0.5,
                   c=[segment_colors[seg_id]], label=segment_names[seg_id])
ax1.set_aspect('equal')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Top View (XY) - All Segments')
ax1.legend(markerscale=10, loc='upper right')

# 2-7. XY views for each segment
for i, (seg_id, name) in enumerate(segment_names.items()):
    ax = fig.add_subplot(3, 4, 2 + i)
    mask = labels == seg_id
    if np.sum(mask) > 100:
        sample = np.random.choice(np.where(mask)[0], min(20000, np.sum(mask)), replace=False)
        ax.scatter(points[sample, 0], points[sample, 1], s=0.5, alpha=0.5, c=[segment_colors[seg_id]])
    # Draw reference circle
    circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{name} ({np.sum(mask):,} pts)')
    ax.set_xlim(-15, -6)
    ax.set_ylim(-5, 4)

# 8. XZ side view colored by segment
ax8 = fig.add_subplot(3, 4, 8)
for seg_id in segment_names.keys():
    mask = labels == seg_id
    if np.sum(mask) > 0:
        sample = np.random.choice(np.where(mask)[0], min(10000, np.sum(mask)), replace=False)
        ax8.scatter(points[sample, 0], points[sample, 2], s=0.5, alpha=0.5,
                   c=[segment_colors[seg_id]], label=segment_names[seg_id])
ax8.set_xlabel('X (m)')
ax8.set_ylabel('Z (m)')
ax8.set_title('Front View (XZ) - All Segments')

# 9-12. XZ views for key segments
key_segments = [1, 2, 3, 4]  # Wall, Floors, Central, Vertical
for i, seg_id in enumerate(key_segments):
    ax = fig.add_subplot(3, 4, 9 + i)
    mask = labels == seg_id
    if np.sum(mask) > 100:
        sample = np.random.choice(np.where(mask)[0], min(20000, np.sum(mask)), replace=False)
        ax.scatter(points[sample, 0], points[sample, 2], s=0.5, alpha=0.5, c=[segment_colors[seg_id]])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'{segment_names[seg_id]} - Side View')

plt.tight_layout()
plt.savefig('visualizations/segmentation_results.png', dpi=150)
print("Saved: visualizations/segmentation_results.png")

# Create a second figure with slice views at different Z levels
fig2 = plt.figure(figsize=(20, 16))

z_slices = [z_min + 0.5, -524, -522, -521.5, -520, -518, -517, z_max - 0.5]
slice_thickness = 0.25

for i, z in enumerate(z_slices):
    ax = fig2.add_subplot(2, 4, i + 1)
    slice_mask = np.abs(points[:, 2] - z) < slice_thickness
    slice_labels = labels[slice_mask]
    slice_pts = points[slice_mask]

    for seg_id in segment_names.keys():
        seg_mask = slice_labels == seg_id
        if np.sum(seg_mask) > 0:
            ax.scatter(slice_pts[seg_mask, 0], slice_pts[seg_mask, 1],
                      s=1, alpha=0.7, c=[segment_colors[seg_id]], label=segment_names[seg_id])

    circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Z = {z:.2f}m ({np.sum(slice_mask):,} pts)')
    ax.set_xlim(-15, -6)
    ax.set_ylim(-5, 4)
    if i == 0:
        ax.legend(markerscale=5, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('visualizations/segmentation_slices.png', dpi=150)
print("Saved: visualizations/segmentation_slices.png")

# Save segmented point cloud
print("\nSaving segmented point cloud...")
output_las = laspy.create(point_format=las.point_format, file_version=las.header.version)
output_las.x = las.x
output_las.y = las.y
output_las.z = las.z
output_las.intensity = las.intensity

# Store segment labels in classification field
output_las.classification = labels.astype(np.uint8)

# Also create RGB colors based on segments
red = np.zeros(len(points), dtype=np.uint16)
green = np.zeros(len(points), dtype=np.uint16)
blue = np.zeros(len(points), dtype=np.uint16)

for seg_id, color in segment_colors.items():
    mask = labels == seg_id
    red[mask] = int(color[0] * 65535)
    green[mask] = int(color[1] * 65535)
    blue[mask] = int(color[2] * 65535)

# Check if we can add RGB
if hasattr(output_las, 'red'):
    output_las.red = red
    output_las.green = green
    output_las.blue = blue

output_las.write('segmented_shaft.las')
print("Saved: segmented_shaft.las")

# Save individual segment point clouds
print("\nSaving individual segment files...")
for seg_id, name in segment_names.items():
    mask = labels == seg_id
    if np.sum(mask) > 0:
        seg_las = laspy.create(point_format=las.point_format, file_version=las.header.version)
        seg_las.x = las.x[mask]
        seg_las.y = las.y[mask]
        seg_las.z = las.z[mask]
        seg_las.intensity = las.intensity[mask]
        filename = f'segments/segment_{seg_id}_{name.lower().replace("/", "_").replace(" ", "_")}.las'
        import os
        os.makedirs('segments', exist_ok=True)
        seg_las.write(filename)
        print(f"  Saved: {filename}")

print("\nSegmentation complete!")
