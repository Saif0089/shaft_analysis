/**
 * Main Application Logic
 */
class App {
    constructor() {
        this.viewer = null;
        this.annotator = null;
        this.currentSlice = null;
        this.labels = [];

        this.init();
    }

    async init() {
        // Initialize viewer
        this.viewer = new PointCloudViewer('viewer-canvas');

        // Initialize annotator
        this.annotator = new Annotator(this.viewer);
        this.annotator.onBoxSelected = (box) => this.onBoxSelected(box);
        this.annotator.onBoxChanged = (box) => this.onBoxChanged(box);

        // Setup UI handlers
        this.setupEventListeners();

        // Load initial data
        await this.loadLabels();
        await this.loadSlices();
        await this.loadModelStatus();

        this.setStatus('Ready');
    }

    setupEventListeners() {
        // Slice controls
        document.getElementById('load-slice-btn').addEventListener('click', () => this.loadSelectedSlice());
        document.getElementById('copy-slice-btn').addEventListener('click', () => this.showCopySliceModal());

        // View controls
        document.getElementById('point-size').addEventListener('input', (e) => {
            const size = parseInt(e.target.value);
            document.getElementById('point-size-value').textContent = size;
            this.viewer.setPointSize(size);
        });

        document.getElementById('color-mode').addEventListener('change', (e) => {
            this.viewer.setColorMode(e.target.value);
        });

        document.getElementById('reset-camera-btn').addEventListener('click', () => this.viewer.resetCamera());
        document.getElementById('top-view-btn').addEventListener('click', () => this.viewer.setTopView());

        // Mode buttons
        document.getElementById('mode-view').addEventListener('click', () => this.setMode('view'));
        document.getElementById('mode-translate').addEventListener('click', () => this.setMode('translate'));
        document.getElementById('mode-rotate').addEventListener('click', () => this.setMode('rotate'));
        document.getElementById('mode-scale').addEventListener('click', () => this.setMode('scale'));

        // Label controls
        document.getElementById('add-label-btn').addEventListener('click', () => this.addLabel());

        // Box controls
        document.getElementById('add-box-btn').addEventListener('click', () => this.annotator.addBox());
        document.getElementById('delete-box-btn').addEventListener('click', () => {
            if (this.annotator.selectedBox) {
                this.annotator.deleteBox(this.annotator.selectedBox);
            }
        });

        // Box property inputs
        ['box-pos-x', 'box-pos-y', 'box-pos-z',
         'box-scale-x', 'box-scale-y', 'box-scale-z',
         'box-rot-x', 'box-rot-y', 'box-rot-z'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => this.updateBoxFromInputs());
        });

        document.getElementById('box-label').addEventListener('change', (e) => {
            const label = this.labels.find(l => l.name === e.target.value);
            if (label) {
                this.annotator.setCurrentLabel(label);
            }
        });

        // Action buttons
        document.getElementById('save-annotations-btn').addEventListener('click', () => this.saveAnnotations());
        document.getElementById('generate-gt-btn').addEventListener('click', () => this.generateGroundTruth());
        document.getElementById('train-btn').addEventListener('click', () => this.trainModel());
        document.getElementById('inference-btn').addEventListener('click', () => this.runInference());
        document.getElementById('submit-corrections-btn').addEventListener('click', () => this.submitCorrections());

        // Copy slice modal
        document.getElementById('copy-confirm-btn').addEventListener('click', () => this.copySelectedSlice());
        document.getElementById('copy-cancel-btn').addEventListener('click', () => this.hideCopySliceModal());
    }

    setMode(mode) {
        this.annotator.setMode(mode);

        // Update button states
        document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById(`mode-${mode}`).classList.add('active');
    }

    async loadSlices() {
        try {
            const response = await fetch('/api/slices');
            const data = await response.json();

            const select = document.getElementById('slice-select');
            select.innerHTML = '<option value="">Select a slice...</option>';

            for (const slice of data.slices) {
                const option = document.createElement('option');
                option.value = slice.name;
                option.textContent = `${slice.name} (Z: ${slice.z_min.toFixed(1)} to ${slice.z_max.toFixed(1)})`;
                if (slice.has_annotation) option.textContent += ' [annotated]';
                if (slice.has_ground_truth) option.textContent += ' [GT]';
                select.appendChild(option);
            }
        } catch (error) {
            console.error('Error loading slices:', error);
            this.setStatus('Error loading slices');
        }
    }

    async loadSelectedSlice() {
        const select = document.getElementById('slice-select');
        const sliceName = select.value;

        if (!sliceName) {
            alert('Please select a slice first');
            return;
        }

        this.showLoading(true);
        this.setStatus(`Loading ${sliceName}...`);

        try {
            // Load point cloud
            const response = await fetch(`/api/slice/${sliceName}`);
            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            this.currentSlice = sliceName;
            this.viewer.loadPointCloud(data);

            // Update UI
            document.getElementById('current-slice').textContent = sliceName;
            document.getElementById('point-count').textContent =
                `${data.num_points.toLocaleString()} / ${data.total_points.toLocaleString()} points`;

            // Load annotations if they exist
            await this.loadAnnotations();

            this.setStatus(`Loaded ${sliceName}`);
        } catch (error) {
            console.error('Error loading slice:', error);
            this.setStatus(`Error: ${error.message}`);
        }

        this.showLoading(false);
    }

    async loadAnnotations() {
        if (!this.currentSlice) return;

        try {
            const response = await fetch(`/api/annotations/${this.currentSlice}`);
            const data = await response.json();

            if (data.boxes && data.boxes.length > 0) {
                this.annotator.loadBoxes(data.boxes);
                this.updateBoxList();
            }
        } catch (error) {
            console.error('Error loading annotations:', error);
        }
    }

    async saveAnnotations() {
        if (!this.currentSlice) {
            alert('No slice loaded');
            return;
        }

        const boxes = this.annotator.getAllBoxesData();

        try {
            const response = await fetch(`/api/annotations/${this.currentSlice}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ boxes })
            });

            const data = await response.json();
            this.setStatus(`Annotations saved (${boxes.length} boxes)`);
        } catch (error) {
            console.error('Error saving annotations:', error);
            this.setStatus('Error saving annotations');
        }
    }

    async loadLabels() {
        try {
            const response = await fetch('/api/labels');
            const data = await response.json();
            this.labels = data.classes;
            this.updateLabelList();
            this.updateLabelSelect();
            this.viewer.setLabels(this.labels);
            this.annotator.setLabels(this.labels);
        } catch (error) {
            console.error('Error loading labels:', error);
        }
    }

    async addLabel() {
        const nameInput = document.getElementById('new-label-name');
        const colorInput = document.getElementById('new-label-color');

        const name = nameInput.value.trim();
        const color = colorInput.value;

        if (!name) {
            alert('Please enter a label name');
            return;
        }

        try {
            const response = await fetch('/api/labels', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'add', name, color })
            });

            const data = await response.json();

            if (data.error) {
                alert(data.error);
                return;
            }

            this.labels = data.classes;
            this.updateLabelList();
            this.updateLabelSelect();
            this.viewer.setLabels(this.labels);
            this.annotator.setLabels(this.labels);

            nameInput.value = '';
        } catch (error) {
            console.error('Error adding label:', error);
        }
    }

    async removeLabel(labelId) {
        if (labelId === 0) {
            alert('Cannot remove the unlabeled class');
            return;
        }

        if (!confirm('Are you sure you want to remove this label?')) {
            return;
        }

        try {
            const response = await fetch('/api/labels', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'remove', id: labelId })
            });

            const data = await response.json();
            this.labels = data.classes;
            this.updateLabelList();
            this.updateLabelSelect();
            this.viewer.setLabels(this.labels);
            this.annotator.setLabels(this.labels);
        } catch (error) {
            console.error('Error removing label:', error);
        }
    }

    updateLabelList() {
        const container = document.getElementById('label-list');
        container.innerHTML = '';

        for (const label of this.labels) {
            const item = document.createElement('div');
            item.className = 'label-item';
            item.innerHTML = `
                <div class="label-color" style="background: ${label.color}"></div>
                <span class="label-name">${label.name}</span>
                ${label.id !== 0 ? `<span class="label-delete" data-id="${label.id}">&times;</span>` : ''}
            `;

            item.addEventListener('click', () => {
                this.annotator.setCurrentLabel(label);
                document.querySelectorAll('.label-item').forEach(el => el.classList.remove('selected'));
                item.classList.add('selected');
            });

            const deleteBtn = item.querySelector('.label-delete');
            if (deleteBtn) {
                deleteBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.removeLabel(label.id);
                });
            }

            container.appendChild(item);
        }
    }

    updateLabelSelect() {
        const select = document.getElementById('box-label');
        select.innerHTML = '';

        for (const label of this.labels) {
            const option = document.createElement('option');
            option.value = label.name;
            option.textContent = label.name;
            select.appendChild(option);
        }
    }

    updateBoxList() {
        const container = document.getElementById('box-list');
        container.innerHTML = '';

        for (const box of this.annotator.boxes) {
            const label = this.labels.find(l => l.name === box.userData.label);
            const color = label ? label.color : '#808080';

            const item = document.createElement('div');
            item.className = 'box-item';
            if (box === this.annotator.selectedBox) {
                item.classList.add('selected');
            }

            item.innerHTML = `
                <div class="box-color" style="background: ${color}"></div>
                <span class="box-label">${box.userData.label}</span>
            `;

            item.addEventListener('click', () => {
                this.annotator.selectBox(box);
            });

            container.appendChild(item);
        }
    }

    onBoxSelected(box) {
        this.updateBoxList();

        const propsPanel = document.getElementById('box-properties');
        const noSelection = document.getElementById('no-selection');

        if (box) {
            propsPanel.classList.remove('hidden');
            noSelection.classList.add('hidden');

            const info = this.annotator.getSelectedBoxInfo();

            document.getElementById('box-label').value = info.label;
            document.getElementById('box-pos-x').value = info.position.x.toFixed(2);
            document.getElementById('box-pos-y').value = info.position.y.toFixed(2);
            document.getElementById('box-pos-z').value = info.position.z.toFixed(2);
            document.getElementById('box-scale-x').value = info.scale.x.toFixed(2);
            document.getElementById('box-scale-y').value = info.scale.y.toFixed(2);
            document.getElementById('box-scale-z').value = info.scale.z.toFixed(2);
            document.getElementById('box-rot-x').value = info.rotation.x.toFixed(1);
            document.getElementById('box-rot-y').value = info.rotation.y.toFixed(1);
            document.getElementById('box-rot-z').value = info.rotation.z.toFixed(1);
        } else {
            propsPanel.classList.add('hidden');
            noSelection.classList.remove('hidden');
        }
    }

    onBoxChanged(box) {
        if (box) {
            const info = this.annotator.getSelectedBoxInfo();

            document.getElementById('box-pos-x').value = info.position.x.toFixed(2);
            document.getElementById('box-pos-y').value = info.position.y.toFixed(2);
            document.getElementById('box-pos-z').value = info.position.z.toFixed(2);
            document.getElementById('box-scale-x').value = info.scale.x.toFixed(2);
            document.getElementById('box-scale-y').value = info.scale.y.toFixed(2);
            document.getElementById('box-scale-z').value = info.scale.z.toFixed(2);
            document.getElementById('box-rot-x').value = info.rotation.x.toFixed(1);
            document.getElementById('box-rot-y').value = info.rotation.y.toFixed(1);
            document.getElementById('box-rot-z').value = info.rotation.z.toFixed(1);
        }

        this.updateBoxList();
    }

    updateBoxFromInputs() {
        const position = {
            x: parseFloat(document.getElementById('box-pos-x').value) || 0,
            y: parseFloat(document.getElementById('box-pos-y').value) || 0,
            z: parseFloat(document.getElementById('box-pos-z').value) || 0
        };

        const rotation = {
            x: parseFloat(document.getElementById('box-rot-x').value) || 0,
            y: parseFloat(document.getElementById('box-rot-y').value) || 0,
            z: parseFloat(document.getElementById('box-rot-z').value) || 0
        };

        const scale = {
            x: parseFloat(document.getElementById('box-scale-x').value) || 1,
            y: parseFloat(document.getElementById('box-scale-y').value) || 1,
            z: parseFloat(document.getElementById('box-scale-z').value) || 1
        };

        this.annotator.updateBoxFromInputs(position, rotation, scale);
    }

    async generateGroundTruth() {
        if (!this.currentSlice) {
            alert('No slice loaded');
            return;
        }

        if (this.annotator.boxes.length === 0) {
            alert('No bounding boxes to generate ground truth from');
            return;
        }

        // Save annotations first
        await this.saveAnnotations();

        this.setStatus('Generating ground truth...');

        try {
            const response = await fetch(`/api/generate-ground-truth/${this.currentSlice}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            let statusMsg = `Ground truth generated: ${data.total_points.toLocaleString()} points\n`;
            for (const [label, count] of Object.entries(data.label_counts)) {
                statusMsg += `  ${label}: ${count.toLocaleString()}\n`;
            }

            this.setStatus(statusMsg);
            alert('Ground truth generated successfully!');
        } catch (error) {
            console.error('Error generating ground truth:', error);
            this.setStatus(`Error: ${error.message}`);
        }
    }

    async trainModel() {
        this.setStatus('Training not yet implemented');
        alert('Training functionality will be implemented in the next phase');
    }

    async runInference() {
        this.setStatus('Inference not yet implemented');
        alert('Inference functionality will be implemented in the next phase');
    }

    async submitCorrections() {
        // Save current annotations as corrections and trigger retraining
        await this.saveAnnotations();
        await this.generateGroundTruth();
        this.setStatus('Corrections submitted. Ready for retraining.');
    }

    async loadModelStatus() {
        try {
            const response = await fetch('/api/model-status');
            const data = await response.json();

            const statusDiv = document.getElementById('model-status');
            if (data.has_model) {
                statusDiv.textContent = `Model available (${data.model_count} checkpoints)`;
            } else {
                statusDiv.textContent = 'No trained model';
            }
        } catch (error) {
            console.error('Error loading model status:', error);
        }
    }

    async showCopySliceModal() {
        const modal = document.getElementById('copy-slice-modal');
        const select = document.getElementById('available-slices-select');

        modal.classList.remove('hidden');
        select.innerHTML = '<option value="">Loading...</option>';

        try {
            const response = await fetch('/api/available-slices');
            const data = await response.json();

            select.innerHTML = '<option value="">Select a slice...</option>';

            for (const slice of data.slices) {
                const option = document.createElement('option');
                option.value = slice.name;
                option.textContent = slice.name;
                if (slice.local_exists) {
                    option.textContent += ' (already copied)';
                    option.disabled = true;
                }
                select.appendChild(option);
            }
        } catch (error) {
            console.error('Error loading available slices:', error);
            select.innerHTML = '<option value="">Error loading slices</option>';
        }
    }

    hideCopySliceModal() {
        document.getElementById('copy-slice-modal').classList.add('hidden');
    }

    async copySelectedSlice() {
        const select = document.getElementById('available-slices-select');
        const sliceName = select.value;

        if (!sliceName) {
            alert('Please select a slice to copy');
            return;
        }

        try {
            const response = await fetch(`/api/copy-slice/${sliceName}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            this.hideCopySliceModal();
            await this.loadSlices();
            this.setStatus(`Copied ${sliceName}`);
        } catch (error) {
            console.error('Error copying slice:', error);
            alert(`Error: ${error.message}`);
        }
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.remove('hidden');
        } else {
            overlay.classList.add('hidden');
        }
    }

    setStatus(message) {
        document.getElementById('status-text').textContent = message;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
