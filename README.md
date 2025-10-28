# ChemiAI

A modular deep learning toolkit for mesh-texture processing, training, and inference. This repository provides main APIs for training and inference, along with auxiliary utilities for preprocessing, checkpointing, data transforms, and more.

## Repository

- **GitHub**: [ChemiAI on GitHub](https://github.com/Cheminova-Project/ChemiAI)
- **Clone**:
```bash
git clone https://github.com/Cheminova-Project/ChemiAI.git
cd ChemiAI
```

## Installation

It is recommended to use a dedicated Python environment (e.g., Conda or venv).

- **Conda (recommended)**:
```bash
conda create -n chemiAI python=3.10 -y
conda activate chemiAI
pip install -r requirements.txt
```

- **Python venv**:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Structure

```text
ChemiAI
│
├── requirements.txt                        # Dependencies
│
├── train.py                                # Main training script
├── Infer.py                                # Inference & auto-slicing
│
├── preprocessing/                          # Mesh & texture preprocessing
│   ├── Mesh_Slicing.py                     # Mesh slicing utilities
│   └── Texture_Extraction.py               # Extract texture patches per face
│
├── integrated_texture_geometry_model.py    # Deep learning model
├── mesh_texture_dataset.py                 # Mesh custom dataset
├── loss.py                                 # Masked cross-entropy loss
│
├── tools/
│   ├── auto_mesh_slicer.py                 # Automatic mesh slicing (grid/adaptive)
│   ├── check_point.py                      # Save/load checkpoints
│   ├── helper.py                           # Optimizer & training helpers
│   ├── downst.py                           # Classifier head
│   ├── mesh_dataset.py                     # Auxiliary dataset class functions
│   └── model_base.py                       # Auxiliary deep learning model functions
│
└── src/
    ├── transforms.py                       # Data transformations
    └── utils/
        ├── distributed.py
        ├── logging.py
        ├── schedulers.py
        └── tensors.py
```

Auxiliary scripts in `tools/` and `src/` support the core APIs and typically do not require user modification.

## Configuration

Example configuration files for training and inference are provided in the repository. For inference with auto-slicing, see for example:
- `config_inference_auto_slice_example.yaml`

These configs define dataset paths, model hyperparameters, transforms, and runtime options. Adjust paths and parameters to match your environment and data.

## Data Preprocessing

Preprocessing operates independently from other modules and prepares datasets for training:

```bash
# Navigate to preprocessing utilities
cd preprocessing

# Example: mesh slicing (adjust args as needed)
python Mesh_Slicing.py --input_mesh /path/to/mesh.obj --out_dir /path/to/output

# Example: texture extraction (adjust args as needed)
python Texture_Extraction.py --input_mesh /path/to/mesh.obj --texture /path/to/texture.png --out_dir /path/to/output
```

The `tools/auto_mesh_slicer.py` utility can also perform automatic slicing (grid/adaptive) during inference; see the Inference section below.

## Training

Use `train.py` with a training configuration file:

```bash
python train.py --config /path/to/train_config.yaml
```

- Uses `integrated_texture_geometry_model.py` for the model architecture
- Loads custom dataset implementations from `mesh_texture_dataset.py` and `tools/mesh_dataset.py`
- Applies losses from `loss.py`
- Supports checkpoint save/load via `tools/check_point.py`

## Inference

Run `Infer.py` with an inference configuration. Auto-slicing can be enabled via config options; the script leverages `tools/auto_mesh_slicer.py` when required.

```bash
python Infer.py --config ./config_inference_auto_slice_example.yaml
```

Outputs are written according to configuration settings (e.g., output directories, prediction formats). Refer to the example config for available options.

## Notes on Models, Datasets, and Utilities

- **Models**: Main integrated architecture is implemented in `integrated_texture_geometry_model.py`. Additional base components live in `tools/model_base.py` and the classifier head in `tools/downst.py`.
- **Datasets**: Custom dataset classes are provided in `mesh_texture_dataset.py` and `tools/mesh_dataset.py`.
- **Transforms & Utils**: Data transformations in `src/transforms.py`; distributed training, logging, schedulers, and tensor utilities reside under `src/utils/`.
- **Checkpoints**: Managed via `tools/check_point.py`.

## Troubleshooting

- Verify your Python version matches the one used to create the environment.
- Ensure paths in your config files point to existing data and checkpoint locations.
- If CUDA is required, confirm the correct PyTorch build and drivers are installed.

## License

This project is part of the Cheminova Project initiative. See the repository for license details.

