"""
Configuration parameters for mine shaft steel structure segmentation.
Tune these based on your specific point cloud characteristics.
"""

# =============================================================================
# Wall Removal Parameters
# =============================================================================
WALL_RADIUS_THRESHOLD = 3.3  # Points beyond this radius from shaft center = wall (meters)
SHAFT_CENTER_AUTO = True      # Auto-detect center via centroid, or specify manually
SHAFT_CENTER_XY = None        # Manual center: (x, y) tuple if SHAFT_CENTER_AUTO=False

# =============================================================================
# Local PCA Parameters
# =============================================================================
K_NEIGHBORS = 30              # Number of neighbors for local PCA computation
MIN_NEIGHBORS = 10            # Minimum neighbors required (skip if fewer)

# =============================================================================
# Geometric Feature Thresholds
# =============================================================================
LINEARITY_THRESHOLD = 0.7     # Linearity > this = linear structure (beam/pipe)
PLANARITY_THRESHOLD = 0.6     # Planarity > this = planar structure (platform)
SCATTERING_THRESHOLD = 0.3    # Scattering > this = volumetric/noise

# =============================================================================
# Vertical Member Detection
# =============================================================================
VERTICAL_DOT_THRESHOLD = 0.85  # |direction · Z| > this = vertical (cos ~32°)
VERTICAL_MIN_HEIGHT = 0.5      # Minimum Z-extent for valid vertical member (meters)
VERTICAL_MAX_RADIUS = 0.2      # Maximum cross-section radius for vertical member

# =============================================================================
# Horizontal Member Detection
# =============================================================================
HORIZONTAL_DOT_THRESHOLD = 0.3  # |direction · Z| < this = horizontal (cos ~72°) - relaxed
HORIZONTAL_MIN_LENGTH = 0.2     # Minimum length for valid horizontal member (meters)
HORIZONTAL_MAX_RADIUS = 0.2     # Maximum cross-section radius for horizontal member
HORIZONTAL_REGION_RADIUS = 0.5  # Larger radius for sparse horizontal points

# Orientation bins for horizontal members (degrees from X-axis)
# Cardinal: 0° (X-aligned) and 90° (Y-aligned)
ORIENTATION_BINS = [0, 90]
ORIENTATION_TOLERANCE = 35.0    # ±tolerance for each bin (degrees) - wider to catch more

# =============================================================================
# Platform/Grating Detection
# =============================================================================
PLATFORM_NORMAL_THRESHOLD = 0.9  # |normal · Z| > this = horizontal platform
PLATFORM_MAX_THICKNESS = 0.1     # Maximum Z-thickness of a platform (meters)
PLATFORM_MIN_AREA = 0.5          # Minimum XY area for valid platform (sq meters)

# =============================================================================
# Region Growing / Clustering
# =============================================================================
REGION_GROW_RADIUS = 0.1         # Search radius for region growing (meters)
MIN_CLUSTER_POINTS = 50          # Minimum points to form a valid cluster

# =============================================================================
# Output Labels
# =============================================================================
LABEL_WALL = 0
LABEL_VERTICAL = 1
LABEL_HORIZONTAL_0 = 2           # Horizontal @ 0° (X-aligned)
LABEL_HORIZONTAL_90 = 3          # Horizontal @ 90° (Y-aligned)
LABEL_PLATFORM = 4
LABEL_UNCLASSIFIED = 5

# Label colors for visualization (RGB 0-255)
LABEL_COLORS = {
    LABEL_WALL: (128, 128, 128),         # Gray
    LABEL_VERTICAL: (255, 0, 0),         # Red
    LABEL_HORIZONTAL_0: (0, 255, 0),     # Green
    LABEL_HORIZONTAL_90: (0, 0, 255),    # Blue
    LABEL_PLATFORM: (255, 255, 0),       # Yellow
    LABEL_UNCLASSIFIED: (255, 128, 0),   # Orange
}

LABEL_NAMES = {
    LABEL_WALL: "wall",
    LABEL_VERTICAL: "vertical",
    LABEL_HORIZONTAL_0: "horizontal_0deg",
    LABEL_HORIZONTAL_90: "horizontal_90deg",
    LABEL_PLATFORM: "platforms",
    LABEL_UNCLASSIFIED: "unclassified",
}
