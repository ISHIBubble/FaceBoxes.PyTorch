# Copilot Instructions for FaceBoxes.PyTorch Project

## Project Overview
This is a **Deep Learning Assignment** implementing **FaceBoxes: A CPU Real-time Face Detector** in PyTorch. The project is a PyTorch reimplementation of the original Caffe-based FaceBoxes model for real-time face detection with high accuracy. This is an academic assignment focused on running the model and testing various modifications to its architecture and training procedures.

## Architecture & Model Details

### Core Model: FaceBoxes
- **Location**: `models/faceboxes.py`
- **Architecture Type**: Single Shot Detector (SSD) variant optimized for CPU inference
- **Input Size**: Fixed at 1024×1024 pixels
- **Classes**: 2 (background + face)
- **Base Network**: Custom lightweight CNN with:
  - **CRelu layers**: Concatenated ReLU activation (`[x, -x]`) for efficient feature extraction
  - **Inception modules**: Multi-scale feature extraction with parallel 1×1 and 3×3 convolutions
  - **Detection layers**: Multi-scale predictions from 3 feature maps

### Network Structure
1. **Rapidly Digested Convolutional Layers (RDCL)**:
   - `conv1`: CRelu(3→24, kernel=7, stride=4) + MaxPool
   - `conv2`: CRelu(48→64, kernel=5, stride=2) + MaxPool
   
2. **Multiple Scale Convolutional Layers (MSCL)**:
   - Three Inception modules (128 channels each)
   - `conv3_1/3_2`: 128→256 (stride 2)
   - `conv4_1/4_2`: 256→256 (stride 2)

3. **Multi-box Head**:
   - Detection at 3 scales with different anchor densities
   - 21 anchors at first layer, 1 anchor each at subsequent layers
   - Parallel localization and classification branches

## Key Components

### Data Pipeline (`data/`)
- **Dataset**: WIDER FACE in VOC format
- **Data Structure**:
  - Images: `data/WIDER_FACE/images/`
  - Annotations: `data/WIDER_FACE/annotations/` (XML format)
  - Image list: `img_list.txt`

- **Data Augmentation** (`data_augment.py`):
  - Random cropping with minimum face size constraint (>16px at training scale)
  - Color distortion (brightness, contrast, saturation, hue)
  - Random horizontal flipping
  - Padding to square
  - Random interpolation methods for resizing

- **Preprocessing**:
  - RGB mean subtraction: `(104, 117, 123)` in BGR order
  - Normalized to [0, 1] for bounding boxes
  - Augmentation applied during training

### Anchor/Prior Box Generation (`layers/functions/prior_box.py`)
- **Multi-scale anchors** with different densities:
  - Scale 1 (stride 32): min_sizes=[32, 64, 128] with dense grid (4×4, 2×2, 1 per location)
  - Scale 2 (stride 64): min_sizes=[256] with 1 anchor per location
  - Scale 3 (stride 128): min_sizes=[512] with 1 anchor per location
- **Dense anchor tiling** for small faces at first scale
- **Variance encoding**: [0.1, 0.2] for location regression

### Loss Function (`layers/modules/multibox_loss.py`)
- **MultiBoxLoss**: Standard SSD loss with modifications
  - **Localization Loss**: Smooth L1 loss on matched boxes
  - **Confidence Loss**: Cross-entropy with hard negative mining (3:1 ratio)
  - **Matching Strategy**: Jaccard overlap threshold = 0.35
  - **Hard GT filtering**: Only matches with overlap ≥ 0.2
  - **Loss weighting**: `L = loc_weight * L_loc + L_conf` where `loc_weight = 2.0`

### Bounding Box Utilities (`utils/box_utils.py`)
- **Encoding/Decoding**: Center-size format with variance normalization
- **Matching**: Bipartite matching with best prior assignment
- **IoU/IoF calculations**: For data augmentation and matching
- **NMS**: Wrapper for CPU/GPU implementations in `utils/nms/`

### NMS (Non-Maximum Suppression) (`utils/nms/`)
- **CPU Implementation**: `cpu_nms.pyx` (Cython)
- **GPU Implementation**: `gpu_nms.pyx` + CUDA kernel (`nms_kernel.cu`)
- **Compilation**: Via `make.sh` using Cython build system
- **Threshold**: Default 0.3 for test-time NMS

## Training Pipeline (`train.py`)

### Training Configuration
- **Optimizer**: SGD with momentum=0.9, weight_decay=5e-4
- **Initial Learning Rate**: 1e-3
- **LR Schedule**: Step decay with gamma=0.1 at epochs 200 and 250
- **Warmup**: Optional warmup for early epochs (currently disabled, warmup_epoch=-1)
- **Max Epochs**: 300
- **Batch Size**: Default 32 (configurable)
- **Multi-GPU**: DataParallel support for multiple GPUs

### Training Process
1. Load WIDER FACE dataset with VOC annotation format
2. Initialize FaceBoxes model in 'train' mode
3. Generate prior boxes once at startup
4. For each iteration:
   - Forward pass through network
   - Compute MultiBoxLoss (localization + classification)
   - Backpropagate and update weights
   - Adjust learning rate based on schedule
5. Save checkpoints every 10 epochs (or every 5 after epoch 200)
6. Save final model as `Final_FaceBoxes.pth`

### Key Training Parameters
```python
img_dim = 1024          # Fixed input size
rgb_mean = (104, 117, 123)  # BGR order
num_classes = 2
loc_weight = 2.0        # Localization loss weight
neg_pos_ratio = 7       # Hard negative mining ratio
overlap_thresh = 0.35   # Matching IoU threshold
```

## Testing/Evaluation Pipeline (`test.py`)

### Test Datasets
Three benchmark datasets with different resize scales:
- **FDDB**: resize=3 (2019 images, ellipse annotations)
- **PASCAL Face**: resize=2.5 (851 images)
- **AFW**: resize=1 (205 images)

### Evaluation Process
1. Load pre-trained weights from `weights/FaceBoxes.pth`
2. For each test image:
   - Resize by dataset-specific scale factor
   - Subtract RGB mean
   - Forward pass → location + confidence predictions
   - Generate prior boxes for image size
   - Decode boxes with variance
   - Apply confidence threshold (default 0.05)
   - NMS with threshold 0.3
   - Keep top 750 detections
3. Save results in dataset-specific format
4. Optional: Visualize detections with `--show_image`

### Test Parameters
```python
confidence_threshold = 0.05  # Min score to keep detection
top_k = 5000                 # Max before NMS
nms_threshold = 0.3          # NMS IoU threshold
keep_top_k = 750            # Max after NMS
vis_thres = 0.5             # Visualization threshold
```

## Configuration (`data/config.py`)
```python
cfg = {
    'name': 'FaceBoxes',
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}
```

## Code Structure & Dependencies

### Module Organization
```
models/          # Network architecture
├── faceboxes.py # Main model definition

data/            # Data loading and augmentation
├── config.py    # Model configuration
├── wider_voc.py # Dataset loader
├── data_augment.py # Augmentation pipeline
└── __init__.py

layers/          # Model components
├── functions/
│   └── prior_box.py    # Anchor generation
└── modules/
    └── multibox_loss.py # Loss computation

utils/           # Utility functions
├── box_utils.py        # Box operations
├── nms_wrapper.py      # NMS interface
├── timer.py           # Timing utilities
└── nms/               # NMS implementations
    ├── cpu_nms.pyx
    ├── gpu_nms.pyx
    └── nms_kernel.cu
```

### Key Dependencies
- **PyTorch** ≥ 1.0.0
- **OpenCV** (cv2): Image I/O and processing
- **NumPy**: Array operations
- **Cython**: For NMS compilation
- **CUDA** (optional): GPU acceleration

### Build System
- **Compilation**: `./make.sh` compiles Cython NMS extensions
- **CUDA Path**: `/usr/local/cuda/` (configurable in make.sh)

## Academic Context & Assignment Goals

### Expected Modifications/Experiments
This is a **deep learning assignment** where students will:
1. **Run baseline model** on WIDER FACE and benchmark datasets
2. **Test architecture modifications**:
   - Different backbone networks
   - Modified Inception modules
   - Alternative activation functions (replace CRelu)
   - Different detection head designs
3. **Experiment with training strategies**:
   - Learning rate schedules
   - Data augmentation techniques
   - Loss function variants
   - Anchor box designs
4. **Evaluate performance**:
   - Compare detection accuracy (mAP)
   - Measure inference speed
   - Analyze failure cases

### Performance Benchmarks
Original paper vs. this implementation:
| Dataset | Original Caffe | PyTorch Implementation |
|---------|----------------|------------------------|
| AFW     | 98.98%        | 98.55%                |
| PASCAL  | 96.77%        | 97.05%                |
| FDDB    | 95.90%        | 96.00%                |

## Coding Guidelines & Conventions

### When Modifying Code
1. **Architecture changes**: Edit `models/faceboxes.py`
   - Maintain 3-scale detection paradigm
   - Keep input size at 1024 unless modifying config
   - Return (loc, conf) tuple from forward pass

2. **Training modifications**: Edit `train.py`
   - Preserve checkpoint saving mechanism
   - Log training metrics (loss_l, loss_c, LR, ETA)
   - Maintain compatibility with multi-GPU training

3. **Data augmentation**: Edit `data/data_augment.py`
   - Keep minimum face size constraint (>16px)
   - Preserve bbox normalization to [0, 1]
   - Maintain aspect ratio constraints

4. **Loss function**: Edit `layers/modules/multibox_loss.py`
   - Keep hard negative mining mechanism
   - Preserve variance-based encoding
   - Maintain N (num_positives) normalization

### Variable Naming Conventions
- `loc` / `loc_data` / `loc_t`: Bounding box locations (x, y, w, h)
- `conf` / `conf_data` / `conf_t`: Class confidences/predictions
- `priors`: Anchor boxes in center-size format
- `targets`: Ground truth boxes + labels
- `img_dim`: Training image dimension (1024)
- `rgb_mean`: Mean values for normalization (BGR order!)

### Important Implementation Details
- **BGR vs RGB**: OpenCV uses BGR order; mean is `(104, 117, 123)` in BGR
- **Box Format**: 
  - Ground truth: `[xmin, ymin, xmax, ymax, label]`
  - Priors/Anchors: `[cx, cy, w, h]` (center-size)
  - Predictions: Encoded offsets, decoded to `[xmin, ymin, xmax, ymax]`
- **GPU Training**: Set `cfg['gpu_train'] = True` for GPU, False for CPU
- **Model Modes**: 'train' (returns raw predictions) vs 'test' (applies softmax)

## Common Tasks & Examples

### 1. Training from Scratch
```bash
python3 train.py --batch_size 32 --lr 1e-3 --max_epoch 300
```

### 2. Resume Training
```bash
python3 train.py --resume_net weights/FaceBoxes_epoch_100.pth --resume_epoch 100
```

### 3. Testing on FDDB
```bash
python3 test.py --trained_model weights/FaceBoxes.pth --dataset FDDB
```

### 4. Visualizing Results
```bash
python3 test.py --dataset PASCAL -s --vis_thres 0.5
```

### 5. CPU-only Testing
```bash
python3 test.py --cpu --dataset AFW
```

## Debugging Tips

### Model Not Loading
- Check for 'module.' prefix in state_dict (DataParallel artifact)
- Use provided `remove_prefix()` function in test.py
- Verify checkpoint compatibility with model definition

### NMS Compilation Errors
- Ensure CUDA path is correct in `make.sh`
- Check Cython installation: `pip install cython`
- Fallback to CPU NMS with `--cpu` flag

### Out of Memory During Training
- Reduce `--batch_size`
- Reduce `--num_workers`
- Use fewer GPUs with `--ngpu`

### Poor Detection Results
- Verify image preprocessing (BGR mean, resize scale)
- Check anchor generation matches training config
- Ensure model is in eval mode: `net.eval()`
- Tune confidence and NMS thresholds

### Data Loading Issues
- Verify WIDER FACE directory structure
- Check `img_list.txt` format: `image_name annotation_name`
- Ensure annotations are in VOC XML format
- Validate bounding box coordinates (xmin < xmax, ymin < ymax)

## Important Files for Modifications

### For Architecture Experiments
- **Primary**: `models/faceboxes.py` - entire model architecture
- **Secondary**: `layers/functions/prior_box.py` - anchor design
- **Config**: `data/config.py` - model hyperparameters

### For Training Experiments
- **Primary**: `train.py` - training loop and schedule
- **Secondary**: `layers/modules/multibox_loss.py` - loss computation
- **Config**: Command-line arguments in `train.py`

### For Data Experiments
- **Primary**: `data/data_augment.py` - augmentation pipeline
- **Secondary**: `data/wider_voc.py` - dataset loading
- **Config**: RGB mean and preprocessing in `preproc` class

### For Evaluation
- **Primary**: `test.py` - inference and evaluation
- **Secondary**: `utils/nms_wrapper.py` - post-processing
- **Config**: Test parameters via command-line arguments

## References & Resources

### Paper
**FaceBoxes: A CPU Real-time Face Detector with High Accuracy**
- Authors: Zhang et al. (2017)
- Conference: IJCB 2017
- arXiv: https://arxiv.org/abs/1708.05234

### Original Implementation
- Caffe version: https://github.com/sfzhang15/FaceBoxes

### Related Architectures
- **SSD**: https://github.com/amdegroot/ssd.pytorch
- **RFBNet**: https://github.com/ruinmessi/RFBNet

### Evaluation Tool
- **face-eval**: https://github.com/sfzhang15/face-eval

## Notes for Copilot Assistance

When helping with this project:
1. **Maintain compatibility** with existing checkpoint format
2. **Preserve multi-scale detection** architecture (3 feature maps)
3. **Keep training reproducible** with fixed random seeds when modifying
4. **Document performance impacts** when suggesting architectural changes
5. **Consider inference speed** - model is designed for CPU real-time detection
6. **Respect academic integrity** - guide students rather than completing assignments
7. **Test on toy data first** before full WIDER FACE training
8. **Validate modifications** don't break prior box generation or loss computation

### When Suggesting Modifications
- Explain **why** the modification might improve performance
- Discuss **trade-offs** (accuracy vs speed, memory vs performance)
- Reference **related work** when applicable
- Suggest **ablation studies** to isolate effects
- Recommend **metrics to track** beyond just accuracy

### Common Student Questions
- "How to change backbone?" → Modify `faceboxes.py`, maintain output scales
- "How to add new augmentation?" → Edit `data_augment.py`, preserve bbox validity
- "How to adjust anchors?" → Modify `config.py` and `prior_box.py` together
- "How to speed up training?" → Multi-GPU, larger batch size, mixed precision
- "How to improve small face detection?" → Dense anchors, better augmentation, focal loss

## Academic Honesty Reminder
This is an educational project. When using Copilot:
- Understand every line of code suggested
- Experiment with different approaches
- Document your modifications and results
- Compare performance with baseline
- Analyze failure cases to gain insights
