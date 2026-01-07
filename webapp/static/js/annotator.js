/**
 * Bounding Box Annotation Tool
 */
class Annotator {
    constructor(viewer) {
        this.viewer = viewer;
        this.scene = viewer.scene;
        this.camera = viewer.camera;
        this.renderer = viewer.renderer;
        this.canvas = viewer.canvas;

        this.boxes = [];
        this.selectedBox = null;
        this.transformControls = null;
        this.mode = 'view';  // 'view', 'translate', 'rotate', 'scale'
        this.labels = [];
        this.currentLabel = null;

        this.onBoxSelected = null;  // Callback
        this.onBoxChanged = null;   // Callback

        this.init();
    }

    init() {
        // Create transform controls
        this.transformControls = new THREE.TransformControls(this.camera, this.canvas);
        this.transformControls.setSpace('local');
        this.transformControls.addEventListener('dragging-changed', (event) => {
            this.viewer.controls.enabled = !event.value;
        });
        this.transformControls.addEventListener('change', () => {
            if (this.selectedBox && this.onBoxChanged) {
                this.onBoxChanged(this.selectedBox);
            }
        });
        this.scene.add(this.transformControls);

        // Raycaster for selection
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        // Click handler
        this.canvas.addEventListener('click', (e) => this.onClick(e));

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
    }

    setLabels(labels) {
        this.labels = labels;
        if (labels.length > 0 && !this.currentLabel) {
            this.currentLabel = labels[0];
        }
    }

    setCurrentLabel(label) {
        this.currentLabel = label;
        // Update selected box label
        if (this.selectedBox) {
            this.selectedBox.userData.label = label.name;
            this.updateBoxColor(this.selectedBox);
            if (this.onBoxChanged) {
                this.onBoxChanged(this.selectedBox);
            }
        }
    }

    setMode(mode) {
        this.mode = mode;

        if (mode === 'view') {
            this.transformControls.detach();
        } else if (this.selectedBox) {
            this.transformControls.attach(this.selectedBox);
            this.transformControls.setMode(mode);
        }
    }

    onClick(event) {
        if (this.mode !== 'view') return;

        // Calculate mouse position
        const rect = this.canvas.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        // Raycast to boxes
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const boxMeshes = this.boxes.map(b => b);
        const intersects = this.raycaster.intersectObjects(boxMeshes);

        if (intersects.length > 0) {
            this.selectBox(intersects[0].object);
        } else {
            this.selectBox(null);
        }
    }

    onKeyDown(event) {
        // Don't handle if typing in input
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') return;

        switch (event.key.toLowerCase()) {
            case 'v':
                this.setMode('view');
                break;
            case 't':
                this.setMode('translate');
                break;
            case 'r':
                this.setMode('rotate');
                break;
            case 's':
                this.setMode('scale');
                break;
            case 'n':
                this.addBox();
                break;
            case 'delete':
            case 'backspace':
                if (this.selectedBox) {
                    this.deleteBox(this.selectedBox);
                }
                event.preventDefault();
                break;
        }
    }

    addBox() {
        // Create box at center of view
        const size = 2;
        const geometry = new THREE.BoxGeometry(size, size, size);
        const material = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.3,
            wireframe: false
        });

        const box = new THREE.Mesh(geometry, material);

        // Add wireframe
        const wireframeGeometry = new THREE.EdgesGeometry(geometry);
        const wireframeMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 });
        const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
        box.add(wireframe);
        box.userData.wireframe = wireframe;

        // Set label
        const label = this.currentLabel || { name: 'unlabeled', color: '#808080' };
        box.userData.id = this.generateId();
        box.userData.label = label.name;
        box.userData.originalColor = label.color;

        this.updateBoxColor(box);

        // Position at camera target
        box.position.copy(this.viewer.controls.target);

        this.scene.add(box);
        this.boxes.push(box);

        // Select the new box
        this.selectBox(box);
        this.setMode('translate');

        return box;
    }

    deleteBox(box) {
        const index = this.boxes.indexOf(box);
        if (index > -1) {
            this.boxes.splice(index, 1);
            this.scene.remove(box);

            if (this.selectedBox === box) {
                this.selectBox(null);
            }

            if (this.onBoxChanged) {
                this.onBoxChanged(null);
            }
        }
    }

    selectBox(box) {
        // Deselect previous
        if (this.selectedBox) {
            this.selectedBox.userData.selected = false;
            this.updateBoxColor(this.selectedBox);
        }

        this.selectedBox = box;

        if (box) {
            box.userData.selected = true;
            this.updateBoxColor(box);

            // Attach transform controls if in transform mode
            if (this.mode !== 'view') {
                this.transformControls.attach(box);
                this.transformControls.setMode(this.mode);
            }
        } else {
            this.transformControls.detach();
        }

        if (this.onBoxSelected) {
            this.onBoxSelected(box);
        }
    }

    updateBoxColor(box) {
        const label = this.labels.find(l => l.name === box.userData.label);
        const colorHex = label ? label.color : '#808080';

        // Parse hex color
        const color = new THREE.Color(colorHex);

        // Update material
        if (box.userData.selected) {
            box.material.color.setHex(0xffff00);
            box.material.opacity = 0.4;
        } else {
            box.material.color.copy(color);
            box.material.opacity = 0.3;
        }

        // Update wireframe
        if (box.userData.wireframe) {
            if (box.userData.selected) {
                box.userData.wireframe.material.color.setHex(0xffff00);
            } else {
                box.userData.wireframe.material.color.copy(color);
            }
        }
    }

    updateBoxFromInputs(position, rotation, scale) {
        if (!this.selectedBox) return;

        this.selectedBox.position.set(position.x, position.y, position.z);
        this.selectedBox.rotation.set(
            rotation.x * Math.PI / 180,
            rotation.y * Math.PI / 180,
            rotation.z * Math.PI / 180
        );
        this.selectedBox.scale.set(scale.x, scale.y, scale.z);

        if (this.onBoxChanged) {
            this.onBoxChanged(this.selectedBox);
        }
    }

    getBoxData(box) {
        // Convert to world coordinates
        const worldPos = this.viewer.sceneToWorld(
            box.position.x,
            box.position.y,
            box.position.z
        );

        return {
            id: box.userData.id,
            label: box.userData.label,
            position: [worldPos.x, worldPos.y, worldPos.z],
            rotation: [box.rotation.x, box.rotation.y, box.rotation.z],
            scale: [box.scale.x, box.scale.y, box.scale.z]
        };
    }

    getAllBoxesData() {
        return this.boxes.map(box => this.getBoxData(box));
    }

    loadBoxes(boxesData) {
        // Clear existing boxes
        this.clearAllBoxes();

        // Create boxes from data
        for (const data of boxesData) {
            const geometry = new THREE.BoxGeometry(1, 1, 1);
            const material = new THREE.MeshBasicMaterial({
                color: 0xff0000,
                transparent: true,
                opacity: 0.3
            });

            const box = new THREE.Mesh(geometry, material);

            // Add wireframe
            const wireframeGeometry = new THREE.EdgesGeometry(geometry);
            const wireframeMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 });
            const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
            box.add(wireframe);
            box.userData.wireframe = wireframe;

            // Set properties
            box.userData.id = data.id || this.generateId();
            box.userData.label = data.label || 'unlabeled';

            // Convert world to scene coordinates
            const scenePos = this.viewer.worldToScene(
                data.position[0],
                data.position[1],
                data.position[2]
            );
            box.position.set(scenePos.x, scenePos.y, scenePos.z);
            box.rotation.set(data.rotation[0], data.rotation[1], data.rotation[2]);
            box.scale.set(data.scale[0], data.scale[1], data.scale[2]);

            this.updateBoxColor(box);

            this.scene.add(box);
            this.boxes.push(box);
        }
    }

    clearAllBoxes() {
        for (const box of this.boxes) {
            this.scene.remove(box);
        }
        this.boxes = [];
        this.selectBox(null);
    }

    generateId() {
        return 'box_' + Math.random().toString(36).substr(2, 9);
    }

    // Get selected box info for UI
    getSelectedBoxInfo() {
        if (!this.selectedBox) return null;

        const worldPos = this.viewer.sceneToWorld(
            this.selectedBox.position.x,
            this.selectedBox.position.y,
            this.selectedBox.position.z
        );

        return {
            id: this.selectedBox.userData.id,
            label: this.selectedBox.userData.label,
            position: {
                x: this.selectedBox.position.x,
                y: this.selectedBox.position.y,
                z: this.selectedBox.position.z
            },
            rotation: {
                x: this.selectedBox.rotation.x * 180 / Math.PI,
                y: this.selectedBox.rotation.y * 180 / Math.PI,
                z: this.selectedBox.rotation.z * 180 / Math.PI
            },
            scale: {
                x: this.selectedBox.scale.x,
                y: this.selectedBox.scale.y,
                z: this.selectedBox.scale.z
            },
            worldPosition: worldPos
        };
    }
}
