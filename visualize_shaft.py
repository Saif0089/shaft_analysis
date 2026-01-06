import laspy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the LAS point cloud
print("Loading LAS point cloud...")
las = laspy.read("../Navvis5mmShaft14_middle_10m.las")

# Extract point coordinates
points = np.vstack([las.x, las.y, las.z]).T
print(f"Point cloud shape: {points.shape}")
print(f"X range: {points[:, 0].min():.3f} to {points[:, 0].max():.3f}")
print(f"Y range: {points[:, 1].min():.3f} to {points[:, 1].max():.3f}")
print(f"Z range: {points[:, 2].min():.3f} to {points[:, 2].max():.3f}")

# Check for colors
if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
    colors = np.vstack([las.red, las.green, las.blue]).T / 65535.0
    print("Colors available in point cloud")
else:
    colors = None
    print("No colors in point cloud")

# Check for intensity
if hasattr(las, 'intensity'):
    intensity = las.intensity
    print(f"Intensity range: {intensity.min()} to {intensity.max()}")
else:
    intensity = None

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

if colors is not None:
    pcd.colors = o3d.utility.Vector3dVector(colors)
elif intensity is not None:
    # Use intensity as grayscale color
    norm_intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    gray_colors = np.column_stack([norm_intensity, norm_intensity, norm_intensity])
    pcd.colors = o3d.utility.Vector3dVector(gray_colors)

# Load the mesh
print("\nLoading OBJ mesh...")
mesh = o3d.io.read_triangle_mesh("../shaft_mesh_hd.obj")
print(f"Mesh vertices: {len(mesh.vertices)}")
print(f"Mesh triangles: {len(mesh.triangles)}")
mesh.compute_vertex_normals()

# Get bounding box info
bbox = pcd.get_axis_aligned_bounding_box()
center = bbox.get_center()
extent = bbox.get_extent()
print(f"\nBounding box center: {center}")
print(f"Bounding box extent: {extent}")

# Create output directory
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

# Function to render from different viewpoints
def render_view(geometry, view_name, look_at, eye, up, width=1920, height=1080):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(geometry)

    # Set view
    ctr = vis.get_view_control()
    ctr.set_lookat(look_at)
    ctr.set_front(np.array(eye) - np.array(look_at))
    ctr.set_up(up)
    ctr.set_zoom(0.5)

    vis.poll_events()
    vis.update_renderer()

    # Capture image
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    return np.asarray(img)

# Define viewpoints based on the data extent
dist = max(extent) * 1.5
views = [
    ("top", center + [0, 0, dist], [0, 1, 0]),
    ("bottom", center + [0, 0, -dist], [0, 1, 0]),
    ("front", center + [0, -dist, 0], [0, 0, 1]),
    ("back", center + [0, dist, 0], [0, 0, 1]),
    ("left", center + [-dist, 0, 0], [0, 0, 1]),
    ("right", center + [dist, 0, 0], [0, 0, 1]),
    ("iso1", center + [dist*0.7, -dist*0.7, dist*0.5], [0, 0, 1]),
    ("iso2", center + [-dist*0.7, -dist*0.7, dist*0.5], [0, 0, 1]),
    ("iso3", center + [dist*0.7, dist*0.7, dist*0.5], [0, 0, 1]),
    ("iso4", center + [-dist*0.7, dist*0.7, dist*0.5], [0, 0, 1]),
]

print("\nRendering point cloud views...")
for view_name, eye, up in views:
    try:
        img = render_view(pcd, view_name, center, eye, up)
        plt.imsave(output_dir / f"pointcloud_{view_name}.png", img)
        print(f"  Saved: pointcloud_{view_name}.png")
    except Exception as e:
        print(f"  Error rendering {view_name}: {e}")

print("\nRendering mesh views...")
for view_name, eye, up in views:
    try:
        img = render_view(mesh, view_name, center, eye, up)
        plt.imsave(output_dir / f"mesh_{view_name}.png", img)
        print(f"  Saved: mesh_{view_name}.png")
    except Exception as e:
        print(f"  Error rendering {view_name}: {e}")

# Create a statistical overview plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Point distribution histograms
axes[0, 0].hist(points[:, 0], bins=100, alpha=0.7)
axes[0, 0].set_title('X Distribution')
axes[0, 0].set_xlabel('X coordinate')

axes[0, 1].hist(points[:, 1], bins=100, alpha=0.7)
axes[0, 1].set_title('Y Distribution')
axes[0, 1].set_xlabel('Y coordinate')

axes[0, 2].hist(points[:, 2], bins=100, alpha=0.7)
axes[0, 2].set_title('Z Distribution')
axes[0, 2].set_xlabel('Z coordinate')

# 2D projections
axes[1, 0].scatter(points[::10, 0], points[::10, 1], s=0.1, alpha=0.5)
axes[1, 0].set_title('XY Projection (Top View)')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')
axes[1, 0].set_aspect('equal')

axes[1, 1].scatter(points[::10, 0], points[::10, 2], s=0.1, alpha=0.5)
axes[1, 1].set_title('XZ Projection (Front View)')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Z')
axes[1, 1].set_aspect('equal')

axes[1, 2].scatter(points[::10, 1], points[::10, 2], s=0.1, alpha=0.5)
axes[1, 2].set_title('YZ Projection (Side View)')
axes[1, 2].set_xlabel('Y')
axes[1, 2].set_ylabel('Z')
axes[1, 2].set_aspect('equal')

plt.tight_layout()
plt.savefig(output_dir / "statistics_overview.png", dpi=150)
print("\nSaved: statistics_overview.png")

print("\nVisualization complete!")
