# Water Segmentation using Enhanced U-Net

This project implements an enhanced U-Net deep learning model for semantic segmentation of water bodies in satellite images. The model is designed to accurately detect and segment water regions for applications in hydrological analysis, environmental monitoring, and remote sensing.

## Repository Contents

- `Water Segmentation using Enhanced U-Net.ipynb`: Jupyter Notebook containing data preprocessing, model architecture, training, and evaluation.
- `final_model.pth`: Trained PyTorch model weights.

## Model Architecture

The model is based on the U-Net architecture with enhancements such as:
- Additional dropout and batch normalization layers
- Modified encoder-decoder structure
- Custom loss functions (e.g., Dice Loss, IoU Loss) to handle class imbalance

## Features

- Preprocessing pipeline for satellite imagery
- Data augmentation during training
- Real-time tracking of IoU and Dice Score metrics
- Visualization of predicted segmentation masks
- Modular and extendable design

## Dependencies

Ensure the following packages are installed:

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn torch torchvision albumentations
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/ahmedmohamedabdelsalam/Water-Segmentation-.git
cd Water-Segmentation-
```

2. Launch the Jupyter Notebook:

```bash
jupyter notebook "Water Segmentation using Enhanced U-Net.ipynb"
```

3. Follow the notebook to load data, train the model, and evaluate results.

## Results

The model's performance is evaluated using:
- Dice Coefficient
- Intersection over Union (IoU)
- Binary Accuracy

## References

- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Publicly available satellite imagery datasets

## To-Do

- Add more diverse training samples
- Implement attention-based U-Net variant
- Deploy the model using Streamlit or Flask

## Contact

For questions or collaborations, please reach out at [ahmedabdelsalam.300200@gmail.com].
