import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# Load the LAS point cloud
print("Loading LAS point cloud...")
las = laspy.read("../Navvis5mmShaft14_middle_10m.las")
points = np.vstack([las.x, las.y, las.z]).T
intensity = las.intensity if hasattr(las, 'intensity') else None

print(f"Total points: {len(points):,}")
print(f"\n{'='*60}")
print("PRECISE MEASUREMENTS")
print('='*60)

# Basic dimensions
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
z_min, z_max = points[:, 2].min(), points[:, 2].max()

print(f"\n--- Bounding Box ---")
print(f"X: {x_min:.4f} to {x_max:.4f} (width: {x_max - x_min:.4f} m)")
print(f"Y: {y_min:.4f} to {y_max:.4f} (depth: {y_max - y_min:.4f} m)")
print(f"Z: {z_min:.4f} to {z_max:.4f} (height: {z_max - z_min:.4f} m)")

# Center of the point cloud
center_x = (x_min + x_max) / 2
center_y = (y_min + y_max) / 2
center_z = (z_min + z_max) / 2
print(f"\n--- Center Point ---")
print(f"Center: ({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")

# Calculate radial distances from center (for cylindrical analysis)
radial_dist = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
print(f"\n--- Radial Analysis (from center) ---")
print(f"Min radius: {radial_dist.min():.4f} m")
print(f"Max radius: {radial_dist.max():.4f} m")
print(f"Mean radius: {radial_dist.mean():.4f} m")
print(f"Median radius: {np.median(radial_dist):.4f} m")

# Find the outer wall radius (using percentile to filter outliers)
outer_radius = np.percentile(radial_dist, 99)
inner_core_radius = np.percentile(radial_dist, 5)
print(f"Outer wall radius (99th percentile): {outer_radius:.4f} m")
print(f"Inner core radius (5th percentile): {inner_core_radius:.4f} m")
print(f"Estimated shaft diameter: {2 * outer_radius:.4f} m")

# Z-level analysis to find floors/platforms
print(f"\n--- Z-Level Analysis (Floor Detection) ---")
z_hist, z_edges = np.histogram(points[:, 2], bins=500)
z_centers = (z_edges[:-1] + z_edges[1:]) / 2
bin_width = z_edges[1] - z_edges[0]
print(f"Z histogram bin width: {bin_width:.4f} m")

# Find peaks in Z distribution (floors have more points)
# Use prominence to find significant peaks
peaks, properties = find_peaks(z_hist, height=np.mean(z_hist)*1.5, distance=10, prominence=500)
floor_z_levels = z_centers[peaks]
floor_z_levels = np.sort(floor_z_levels)

print(f"\nDetected {len(floor_z_levels)} floor/platform levels:")
for i, z in enumerate(floor_z_levels):
    points_at_level = np.sum(np.abs(points[:, 2] - z) < 0.05)
    print(f"  Level {i+1}: Z = {z:.4f} m ({points_at_level:,} points within 5cm)")

if len(floor_z_levels) > 1:
    floor_spacings = np.diff(floor_z_levels)
    print(f"\nFloor spacing analysis:")
    print(f"  Min spacing: {floor_spacings.min():.4f} m")
    print(f"  Max spacing: {floor_spacings.max():.4f} m")
    print(f"  Mean spacing: {floor_spacings.mean():.4f} m")
    print(f"  Std dev: {floor_spacings.std():.4f} m")

# Point density analysis
print(f"\n--- Point Density Analysis ---")
volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
density = len(points) / volume
print(f"Overall density: {density:.2f} points/m³")

# Local density analysis using grid
grid_size = 0.1  # 10cm grid
print(f"Grid cell size for local density: {grid_size} m")

# Analyze horizontal slices
slice_thickness = 0.1
print(f"\n--- Horizontal Slice Analysis ---")
slice_z_values = np.arange(z_min + 0.5, z_max - 0.5, 1.0)  # Every 1m
for z in slice_z_values[:5]:  # First 5 slices
    mask = np.abs(points[:, 2] - z) < slice_thickness/2
    slice_points = points[mask]
    if len(slice_points) > 100:
        slice_radii = np.sqrt((slice_points[:, 0] - center_x)**2 + (slice_points[:, 1] - center_y)**2)
        print(f"  Z={z:.2f}m: {len(slice_points):,} pts, radius range: {slice_radii.min():.3f}-{slice_radii.max():.3f}m")

# Nearest neighbor analysis for point spacing
print(f"\n--- Point Spacing Analysis (sampling) ---")
sample_idx = np.random.choice(len(points), min(10000, len(points)), replace=False)
sample_points = points[sample_idx]

# Quick KNN for spacing estimate
from scipy.spatial import cKDTree
tree = cKDTree(sample_points)
distances, _ = tree.query(sample_points, k=2)  # k=2 because first neighbor is itself
nn_distances = distances[:, 1]
print(f"Nearest neighbor distance (sample of 10k points):")
print(f"  Min: {nn_distances.min():.6f} m ({nn_distances.min()*1000:.3f} mm)")
print(f"  Max: {nn_distances.max():.6f} m ({nn_distances.max()*1000:.3f} mm)")
print(f"  Mean: {nn_distances.mean():.6f} m ({nn_distances.mean()*1000:.3f} mm)")
print(f"  Median: {np.median(nn_distances):.6f} m ({np.median(nn_distances)*1000:.3f} mm)")
print(f"  Std: {nn_distances.std():.6f} m ({nn_distances.std()*1000:.3f} mm)")

# Analyze wall thickness (radial histogram)
print(f"\n--- Wall/Structure Thickness Analysis ---")
radial_hist, radial_edges = np.histogram(radial_dist, bins=100)
radial_centers = (radial_edges[:-1] + radial_edges[1:]) / 2
radial_bin_width = radial_edges[1] - radial_edges[0]
print(f"Radial bin width: {radial_bin_width:.4f} m")

# Find peaks in radial distribution (structural elements)
radial_peaks, _ = find_peaks(radial_hist, height=np.mean(radial_hist), prominence=100)
if len(radial_peaks) > 0:
    print(f"Radial structure peaks at distances:")
    for r in radial_centers[radial_peaks]:
        print(f"  r = {r:.4f} m")

# Intensity analysis if available
if intensity is not None:
    print(f"\n--- Intensity Analysis ---")
    print(f"Intensity min: {intensity.min()}")
    print(f"Intensity max: {intensity.max()}")
    print(f"Intensity mean: {intensity.mean():.2f}")
    print(f"Intensity std: {intensity.std():.2f}")

    # Intensity percentiles
    percentiles = [10, 25, 50, 75, 90]
    print("Intensity percentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(intensity, p):.0f}")

# Recommended segmentation parameters
print(f"\n{'='*60}")
print("RECOMMENDED SEGMENTATION PARAMETERS")
print('='*60)
print(f"Based on the measurements above:")
print(f"\n1. DBSCAN / Clustering:")
print(f"   eps (neighborhood radius): {nn_distances.mean() * 3:.4f} m (3x mean NN distance)")
print(f"   min_samples: 10-50 (adjust based on noise)")
print(f"\n2. Region Growing:")
print(f"   seed_point_threshold: {nn_distances.mean() * 2:.4f} m")
print(f"   smoothness_threshold: 5-15 degrees")
print(f"\n3. Plane Segmentation (RANSAC):")
print(f"   distance_threshold: {nn_distances.mean() * 2:.4f} m")
print(f"   min_points_per_plane: {int(len(points) * 0.01)} (1% of points)")
print(f"\n4. Voxel Grid Downsampling:")
print(f"   voxel_size: {nn_distances.mean() * 2:.4f} m (2x mean NN distance)")
print(f"   alternative: 0.01 m (1cm) for high detail")
print(f"   alternative: 0.05 m (5cm) for faster processing")
print(f"\n5. Floor/Level Segmentation:")
print(f"   Z-slice thickness: {floor_spacings.mean() * 0.1:.4f} m" if len(floor_z_levels) > 1 else "   Z-slice thickness: 0.1 m")
print(f"   Floor Z-levels: {[f'{z:.3f}' for z in floor_z_levels]}")

# Create detailed visualization
fig = plt.figure(figsize=(20, 16))

# 1. Z histogram with floor detection
ax1 = fig.add_subplot(3, 3, 1)
ax1.barh(z_centers, z_hist, height=bin_width, alpha=0.7)
for z in floor_z_levels:
    ax1.axhline(y=z, color='r', linestyle='--', linewidth=1, alpha=0.7)
ax1.set_xlabel('Point Count')
ax1.set_ylabel('Z (m)')
ax1.set_title('Z Distribution with Detected Floors (red)')

# 2. Radial histogram
ax2 = fig.add_subplot(3, 3, 2)
ax2.bar(radial_centers, radial_hist, width=radial_bin_width, alpha=0.7)
ax2.set_xlabel('Radial Distance from Center (m)')
ax2.set_ylabel('Point Count')
ax2.set_title('Radial Distribution')
ax2.axvline(x=outer_radius, color='r', linestyle='--', label=f'Outer wall: {outer_radius:.2f}m')
ax2.legend()

# 3. NN distance histogram
ax3 = fig.add_subplot(3, 3, 3)
ax3.hist(nn_distances * 1000, bins=50, alpha=0.7)
ax3.set_xlabel('Nearest Neighbor Distance (mm)')
ax3.set_ylabel('Count')
ax3.set_title(f'Point Spacing (mean: {nn_distances.mean()*1000:.2f}mm)')

# 4-6. XY views at different Z levels
z_slices = [z_min + 1, (z_min + z_max)/2, z_max - 1]
for i, z in enumerate(z_slices):
    ax = fig.add_subplot(3, 3, 4 + i)
    mask = np.abs(points[:, 2] - z) < 0.2
    slice_pts = points[mask]
    ax.scatter(slice_pts[:, 0], slice_pts[:, 1], s=0.1, alpha=0.5)
    circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='r', linestyle='--')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'XY Slice at Z={z:.2f}m')

# 7. XZ projection colored by Y
ax7 = fig.add_subplot(3, 3, 7)
idx = np.random.choice(len(points), min(50000, len(points)), replace=False)
scatter = ax7.scatter(points[idx, 0], points[idx, 2], c=points[idx, 1], s=0.1, alpha=0.5, cmap='viridis')
plt.colorbar(scatter, ax=ax7, label='Y (m)')
ax7.set_xlabel('X (m)')
ax7.set_ylabel('Z (m)')
ax7.set_title('XZ Projection (colored by Y)')

# 8. YZ projection colored by X
ax8 = fig.add_subplot(3, 3, 8)
scatter = ax8.scatter(points[idx, 1], points[idx, 2], c=points[idx, 0], s=0.1, alpha=0.5, cmap='viridis')
plt.colorbar(scatter, ax=ax8, label='X (m)')
ax8.set_xlabel('Y (m)')
ax8.set_ylabel('Z (m)')
ax8.set_title('YZ Projection (colored by X)')

# 9. Intensity distribution if available
ax9 = fig.add_subplot(3, 3, 9)
if intensity is not None:
    ax9.hist(intensity, bins=100, alpha=0.7)
    ax9.set_xlabel('Intensity')
    ax9.set_ylabel('Count')
    ax9.set_title('Intensity Distribution')
else:
    ax9.text(0.5, 0.5, 'No intensity data', ha='center', va='center', transform=ax9.transAxes)

plt.tight_layout()
plt.savefig('visualizations/detailed_measurements.png', dpi=150)
print(f"\nSaved: visualizations/detailed_measurements.png")

# Save measurements to file
with open('visualizations/measurements.txt', 'w') as f:
    f.write("SHAFT POINT CLOUD MEASUREMENTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total points: {len(points):,}\n")
    f.write(f"Bounding box:\n")
    f.write(f"  X: {x_min:.4f} to {x_max:.4f} (width: {x_max - x_min:.4f} m)\n")
    f.write(f"  Y: {y_min:.4f} to {y_max:.4f} (depth: {y_max - y_min:.4f} m)\n")
    f.write(f"  Z: {z_min:.4f} to {z_max:.4f} (height: {z_max - z_min:.4f} m)\n")
    f.write(f"\nCenter: ({center_x:.4f}, {center_y:.4f}, {center_z:.4f})\n")
    f.write(f"Estimated shaft diameter: {2 * outer_radius:.4f} m\n")
    f.write(f"Outer wall radius: {outer_radius:.4f} m\n")
    f.write(f"\nFloor Z-levels:\n")
    for i, z in enumerate(floor_z_levels):
        f.write(f"  Level {i+1}: Z = {z:.4f} m\n")
    f.write(f"\nPoint spacing (mean NN distance): {nn_distances.mean():.6f} m ({nn_distances.mean()*1000:.3f} mm)\n")

print("Saved: visualizations/measurements.txt")
