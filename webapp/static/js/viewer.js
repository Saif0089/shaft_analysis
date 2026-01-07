/**
 * Point Cloud Viewer using Three.js
 */
class PointCloudViewer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.pointCloud = null;
        this.pointsData = null;
        this.colorMode = 'height';
        this.pointSize = 2;
        this.labels = [];
        this.predictions = null;

        this.init();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a15);

        // Camera
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
        this.camera.position.set(20, 20, 20);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true
        });
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Orbit Controls
        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = true;
        this.controls.minDistance = 1;
        this.controls.maxDistance = 200;

        // Grid helper
        this.gridHelper = new THREE.GridHelper(50, 50, 0x333333, 0x222222);
        this.scene.add(this.gridHelper);

        // Axes helper
        this.axesHelper = new THREE.AxesHelper(5);
        this.scene.add(this.axesHelper);

        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        // Handle resize
        window.addEventListener('resize', () => this.onResize());

        // Start render loop
        this.animate();
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onResize() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    loadPointCloud(data) {
        // Remove existing point cloud
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }

        this.pointsData = data;

        // Create geometry
        const geometry = new THREE.BufferGeometry();

        // Calculate center for normalization
        const xMin = Math.min(...data.x);
        const xMax = Math.max(...data.x);
        const yMin = Math.min(...data.y);
        const yMax = Math.max(...data.y);
        const zMin = Math.min(...data.z);
        const zMax = Math.max(...data.z);

        const centerX = (xMin + xMax) / 2;
        const centerY = (yMin + yMax) / 2;
        const centerZ = (zMin + zMax) / 2;

        // Store bounds for later use
        this.bounds = {
            x: [xMin, xMax],
            y: [yMin, yMax],
            z: [zMin, zMax],
            center: [centerX, centerY, centerZ]
        };

        // Create position array (centered)
        const positions = new Float32Array(data.num_points * 3);
        for (let i = 0; i < data.num_points; i++) {
            positions[i * 3] = data.x[i] - centerX;
            positions[i * 3 + 1] = data.z[i] - centerZ;  // Swap Y and Z for Three.js
            positions[i * 3 + 2] = data.y[i] - centerY;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Create colors
        this.updateColors();

        // Create material
        const material = new THREE.PointsMaterial({
            size: this.pointSize * 0.01,
            vertexColors: true,
            sizeAttenuation: true
        });

        // Create points
        this.pointCloud = new THREE.Points(geometry, material);
        this.scene.add(this.pointCloud);

        // Update grid position
        this.gridHelper.position.y = (zMin - centerZ);

        // Reset camera
        this.resetCamera();

        return this.bounds;
    }

    updateColors() {
        if (!this.pointsData || !this.pointCloud) return;

        const data = this.pointsData;
        const colors = new Float32Array(data.num_points * 3);

        if (this.colorMode === 'height') {
            // Color by height (Z)
            const zMin = Math.min(...data.z);
            const zMax = Math.max(...data.z);
            const zRange = zMax - zMin || 1;

            for (let i = 0; i < data.num_points; i++) {
                const t = (data.z[i] - zMin) / zRange;
                // Rainbow colormap
                const hue = (1 - t) * 0.7;  // Blue to red
                const color = new THREE.Color().setHSL(hue, 0.8, 0.5);
                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
            }
        } else if (this.colorMode === 'rgb' && data.r) {
            // Use original RGB
            for (let i = 0; i < data.num_points; i++) {
                colors[i * 3] = data.r[i] / 255;
                colors[i * 3 + 1] = data.g[i] / 255;
                colors[i * 3 + 2] = data.b[i] / 255;
            }
        } else if (this.colorMode === 'label' && data.classification) {
            // Color by label
            for (let i = 0; i < data.num_points; i++) {
                const labelId = data.classification[i];
                const color = this.getLabelColor(labelId);
                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
            }
        } else if (this.colorMode === 'prediction' && this.predictions) {
            // Color by prediction
            for (let i = 0; i < data.num_points; i++) {
                const labelId = this.predictions[i] || 0;
                const color = this.getLabelColor(labelId);
                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
            }
        } else {
            // Default gray
            for (let i = 0; i < data.num_points; i++) {
                colors[i * 3] = 0.5;
                colors[i * 3 + 1] = 0.5;
                colors[i * 3 + 2] = 0.5;
            }
        }

        this.pointCloud.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }

    getLabelColor(labelId) {
        const label = this.labels.find(l => l.id === labelId);
        if (label) {
            const hex = label.color.replace('#', '');
            return new THREE.Color(
                parseInt(hex.substr(0, 2), 16) / 255,
                parseInt(hex.substr(2, 2), 16) / 255,
                parseInt(hex.substr(4, 2), 16) / 255
            );
        }
        return new THREE.Color(0.5, 0.5, 0.5);
    }

    setLabels(labels) {
        this.labels = labels;
        if (this.colorMode === 'label' || this.colorMode === 'prediction') {
            this.updateColors();
        }
    }

    setPredictions(predictions) {
        this.predictions = predictions;
        if (this.colorMode === 'prediction') {
            this.updateColors();
        }
    }

    setColorMode(mode) {
        this.colorMode = mode;
        this.updateColors();
    }

    setPointSize(size) {
        this.pointSize = size;
        if (this.pointCloud) {
            this.pointCloud.material.size = size * 0.01;
        }
    }

    resetCamera() {
        if (!this.bounds) return;

        const size = Math.max(
            this.bounds.x[1] - this.bounds.x[0],
            this.bounds.y[1] - this.bounds.y[0],
            this.bounds.z[1] - this.bounds.z[0]
        );

        this.camera.position.set(size, size, size);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    setTopView() {
        if (!this.bounds) return;

        const size = Math.max(
            this.bounds.x[1] - this.bounds.x[0],
            this.bounds.y[1] - this.bounds.y[0]
        );

        this.camera.position.set(0, size * 1.5, 0);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    // Get bounds in world coordinates (for creating boxes)
    getWorldBounds() {
        return this.bounds;
    }

    // Convert world position to scene position (centered)
    worldToScene(x, y, z) {
        if (!this.bounds) return { x: 0, y: 0, z: 0 };
        return {
            x: x - this.bounds.center[0],
            y: z - this.bounds.center[2],  // Swap Y and Z
            z: y - this.bounds.center[1]
        };
    }

    // Convert scene position to world position
    sceneToWorld(x, y, z) {
        if (!this.bounds) return { x: 0, y: 0, z: 0 };
        return {
            x: x + this.bounds.center[0],
            y: z + this.bounds.center[1],  // Swap Y and Z
            z: y + this.bounds.center[2]
        };
    }
}
