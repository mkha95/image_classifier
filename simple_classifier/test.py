import sys
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = resnet50(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with open('imagenet_classes.txt', 'r') as f:
            self.categories = [s.strip() for s in f.readlines()]

    def initUI(self):
        self.setWindowTitle('ResNet50 Image Classifier')
        layout = QVBoxLayout()

        self.upload_btn = QPushButton('Upload Image')
        self.upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_btn)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.result_label = QLabel('Upload an image to see the classification result')
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.setLayout(layout)
        self.setGeometry(300, 300, 400, 400)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', r"<default dir>",
                                                   "Image files (*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp)")
        if file_name:
            try:
                image = Image.open(file_name).convert('RGB')
                pixmap = QPixmap(file_name)
                self.image_label.setPixmap(pixmap.scaled(224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation))

                input_tensor = self.transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = self.model(input_tensor)

                _, predicted_idx = torch.max(output, 1)
                predicted_label = self.categories[predicted_idx.item()]

                self.result_label.setText(f'Predicted class: {predicted_label}')
            except Exception as e:
                self.result_label.setText(f'Error processing image: {str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageClassifierGUI()
    ex.show()
    sys.exit(app.exec_())
