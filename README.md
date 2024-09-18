# Image Classifier Project

This project contains two PyQt5-based image classifiers using ResNet50: a simple classifier and a grid classifier.

## Features

- Simple Image Classifier: Classifies a single image using ResNet50.
- Grid Image Classifier: Divides an image into a grid and classifies each cell.

## Requirements

- Python 3.7+
- PyQt5
- PyTorch
- torchvision
- Pillow

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/mkha95/image_classifier.git
   cd image_classifier
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Simple Classifier

1. Navigate to the simple_classifier directory:
   ```
   cd simple_classifier
   ```

2. Run the script:
   ```
   python test.py
   ```

3. Click "Upload Image" to select an image for classification.

### Grid Classifier

1. Navigate to the grid_classifier directory:
   ```
   cd grid_classifier
   ```

2. Run the script:
   ```
   python test.py
   ```

3. Click "Upload Grid Image" to select an image for classification.
4. Enter the grid size when prompted (e.g., 3x3).
5. Use the "Previous" and "Next" buttons to navigate through the classified grid cells.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project uses the ResNet50 model from torchvision.
- The ImageNet class labels are used for classification.
