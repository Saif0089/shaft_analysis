import laspy
import numpy as np
import matplotlib.pyplot as plt

# Load the segmented point cloud
print("Loading segmented point cloud...")
las = laspy.read("segmented_shaft_final.las")
points = np.vstack([las.x, las.y, las.z]).T
labels = las.classification

# Measurements
center_x, center_y = -10.5549, -0.5347
outer_radius = 3.79

# Radial distance
radial_dist = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)

# Get vertical structures (label 5)
vert_mask = labels == 5
vert_points = points[vert_mask]
vert_radii = radial_dist[vert_mask]

print(f"Total vertical structure points: {np.sum(vert_mask):,}")
print(f"Radial distance range: {vert_radii.min():.3f} - {vert_radii.max():.3f} m")
print(f"Mean radius: {vert_radii.mean():.3f} m")
print(f"Median radius: {np.median(vert_radii):.3f} m")

# Analyze distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Radial histogram
ax = axes[0, 0]
ax.hist(vert_radii, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(x=2.5, color='r', linestyle='--', label='Potential split at 2.5m')
ax.axvline(x=3.0, color='g', linestyle='--', label='Potential split at 3.0m')
ax.set_xlabel('Radial Distance (m)')
ax.set_ylabel('Point Count')
ax.set_title('Vertical Structures - Radial Distribution')
ax.legend()

# 2. Z histogram
ax = axes[0, 1]
ax.hist(vert_points[:, 2], bins=50, alpha=0.7, edgecolor='black')
ax.set_xlabel('Z (m)')
ax.set_ylabel('Point Count')
ax.set_title('Vertical Structures - Z Distribution')

# 3. XY scatter colored by radius
ax = axes[0, 2]
sample = np.random.choice(len(vert_points), min(30000, len(vert_points)), replace=False)
scatter = ax.scatter(vert_points[sample, 0], vert_points[sample, 1],
                     c=vert_radii[sample], s=0.5, alpha=0.5, cmap='viridis')
circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='red', linestyle='--')
ax.add_patch(circle)
circle2 = plt.Circle((center_x, center_y), 2.5, fill=False, color='orange', linestyle='--')
ax.add_patch(circle2)
ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Vertical Structures - Top View (colored by radius)')
plt.colorbar(scatter, ax=ax, label='Radius (m)')

# 4. XZ side view colored by radius
ax = axes[1, 0]
scatter = ax.scatter(vert_points[sample, 0], vert_points[sample, 2],
                     c=vert_radii[sample], s=0.5, alpha=0.5, cmap='viridis')
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Vertical Structures - Side View (colored by radius)')
plt.colorbar(scatter, ax=ax, label='Radius (m)')

# 5. Split preview: inner vs outer
ax = axes[1, 1]
split_radius = 2.8  # Try this split point
inner_mask = vert_radii < split_radius
outer_mask = vert_radii >= split_radius
inner_sample = np.random.choice(np.where(inner_mask)[0], min(15000, np.sum(inner_mask)), replace=False)
outer_sample = np.random.choice(np.where(outer_mask)[0], min(15000, np.sum(outer_mask)), replace=False)
ax.scatter(vert_points[outer_sample, 0], vert_points[outer_sample, 1], s=0.5, alpha=0.5, c='blue', label=f'Outer (r>{split_radius}m) - Pipes/Wires')
ax.scatter(vert_points[inner_sample, 0], vert_points[inner_sample, 1], s=0.5, alpha=0.5, c='red', label=f'Inner (r<{split_radius}m) - Guards')
circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='gray', linestyle='--')
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Split at r={split_radius}m - Top View')
ax.legend(markerscale=10)

# 6. Bottom region analysis (Z < -524)
ax = axes[1, 2]
bottom_mask = vert_points[:, 2] < -524
bottom_pts = vert_points[bottom_mask]
bottom_radii = vert_radii[bottom_mask]
if len(bottom_pts) > 0:
    ax.scatter(bottom_pts[:, 0], bottom_pts[:, 1], c=bottom_pts[:, 2], s=1, alpha=0.7, cmap='coolwarm')
    circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title(f'Bottom Region (Z < -524m) - {len(bottom_pts):,} pts')
else:
    ax.text(0.5, 0.5, 'No points in bottom region', ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')

plt.tight_layout()
plt.savefig('visualizations/vertical_analysis.png', dpi=150)
print("\nSaved: visualizations/vertical_analysis.png")

# Print statistics for different radius bands
print("\n--- Radius Band Analysis ---")
bands = [(0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)]
for r_min, r_max in bands:
    mask = (vert_radii >= r_min) & (vert_radii < r_max)
    count = np.sum(mask)
    if count > 0:
        z_range = f"Z: {vert_points[mask, 2].min():.2f} to {vert_points[mask, 2].max():.2f}"
    else:
        z_range = "N/A"
    print(f"  r={r_min:.1f}-{r_max:.1f}m: {count:,} points ({100*count/len(vert_radii):.1f}%) - {z_range}")

# Analyze the bottom anomaly
print("\n--- Bottom Region Analysis (Z < -524m) ---")
bottom_vert_mask = (labels == 5) & (points[:, 2] < -524)
print(f"Points in bottom region: {np.sum(bottom_vert_mask):,}")
if np.sum(bottom_vert_mask) > 0:
    bottom_radii = radial_dist[bottom_vert_mask]
    print(f"Radial range: {bottom_radii.min():.3f} - {bottom_radii.max():.3f} m")
