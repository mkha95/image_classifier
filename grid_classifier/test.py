import sys
import time
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QInputDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from collections import Counter

class GridImageClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = resnet50(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Load ImageNet class labels
        with open('imagenet_classes.txt', 'r') as f:
            self.categories = [s.strip() for s in f.readlines()]
        self.grid_results = []
        self.current_cell = 0

    def initUI(self):
        self.setWindowTitle('ResNet50 Grid Image Classifier')
        layout = QVBoxLayout()

        self.upload_btn = QPushButton('Upload Grid Image')
        self.upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_btn)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.result_label = QLabel('Upload a grid image to see the classification results')
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.time_label)

        button_layout = QHBoxLayout()
        self.prev_btn = QPushButton('Previous')
        self.prev_btn.clicked.connect(self.show_previous)
        self.prev_btn.setEnabled(False)
        button_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton('Next')
        self.next_btn.clicked.connect(self.show_next)
        self.next_btn.setEnabled(False)
        button_layout.addWidget(self.next_btn)

        layout.addLayout(button_layout)

        self.cell_image_label = QLabel()
        self.cell_image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.cell_image_label)

        self.cell_label = QLabel()
        self.cell_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.cell_label)

        self.setLayout(layout)
        self.setGeometry(300, 300, 500, 700)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', r"<default dir>",
                                                   "Image files (*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp)")
        if file_name:
            try:
                # Open and display the image
                image = Image.open(file_name).convert('RGB')
                pixmap = QPixmap(file_name)
                self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

                # Get grid dimensions from user
                grid_size, ok = QInputDialog.getText(self, 'Input Grid Size', 'Enter grid size (e.g., 3x3):')
                if ok and grid_size:
                    rows, cols = map(int, grid_size.lower().split('x'))

                    # Perform grid classification with timing
                    start_time = time.time()
                    self.grid_results = self.classify_grid(image, rows, cols)
                    end_time = time.time()

                    # Calculate and display classification time
                    classification_time = end_time - start_time
                    self.time_label.setText(f"Classification Time: {classification_time:.2f} seconds")

                    # Count categories
                    category_counts = Counter(result[1] for result in self.grid_results)

                    # Display results
                    result_text = "Classification Results:\n\n"
                    for category, count in category_counts.most_common():
                        result_text += f"{category}: {count}\n"
                    self.result_label.setText(result_text)

                    # Enable navigation buttons
                    self.current_cell = 0
                    self.prev_btn.setEnabled(False)
                    self.next_btn.setEnabled(True)
                    self.show_current_cell()

            except Exception as e:
                self.result_label.setText(f'Error processing image: {str(e)}')

    def classify_grid(self, image, rows, cols):
        width, height = image.size
        cell_width, cell_height = width // cols, height // rows
        results = []
        for i in range(rows):
            for j in range(cols):
                left = j * cell_width
                top = i * cell_height
                right = left + cell_width
                bottom = top + cell_height
                cell = image.crop((left, top, right, bottom))
                input_tensor = self.transform(cell).unsqueeze(0)
                with torch.no_grad():
                    output = self.model(input_tensor)
                _, predicted_idx = torch.max(output, 1)
                predicted_label = self.categories[predicted_idx.item()]
                results.append((cell, predicted_label))
        return results

    def show_current_cell(self):
        cell, label = self.grid_results[self.current_cell]
        pixmap = QPixmap.fromImage(cell.toqimage())
        self.cell_image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.cell_label.setText(f"Cell {self.current_cell + 1}: {label}")

    def show_previous(self):
        if self.current_cell > 0:
            self.current_cell -= 1
            self.show_current_cell()
            self.next_btn.setEnabled(True)
            if self.current_cell == 0:
                self.prev_btn.setEnabled(False)

    def show_next(self):
        if self.current_cell < len(self.grid_results) - 1:
            self.current_cell += 1
            self.show_current_cell()
            self.prev_btn.setEnabled(True)
            if self.current_cell == len(self.grid_results) - 1:
                self.next_btn.setEnabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GridImageClassifierGUI()
    ex.show()
    sys.exit(app.exec_())
