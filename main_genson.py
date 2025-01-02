import sys
import os
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

import controller

class ImageUploader(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Uploader')
        self.setGeometry(100, 100, 800, 600)

        # 라벨 설정
        self.label = QLabel('Click "Upload Image" to select an image', self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet('QLabel { border: 2px dashed #aaa; }')

        # 버튼 설정
        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.clicked.connect(self.upload_image)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.upload_button)
        self.setLayout(layout)

        # 결과 폴더 설정
        self.result_folder = os.path.join(os.getcwd(), 'result')
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            
            self.process_and_save_image(file_path)

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label.setText('')

    def process_and_save_image(self, file_path):
        print(file_path)
        img_name, save_path = controller.img_path(file_path)
        print(img_name)
        # original image save
        pil_image = Image.open(file_path)
        pil_image.convert("RGB").save(save_path+'1-original.jpg')
        origin_img = np.array(pil_image)
        
        if origin_img.ndim == 2:  # Grayscale
            origin_img = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2BGR)
        elif origin_img.shape[2] == 4:  # RGBA
            origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGBA2RGB)
        else:  # RGB
            origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)

        # origin_img = cv2.imread(file_path)
        # save_img = Image.fromarray(origin_img)
        # save_img.save(save_path+'1-original.jpg')

        # image binary
        bin_img = controller.img_binary(origin_img)
        save_img = Image.fromarray(bin_img)
        save_img.save(save_path+'2-binary.jpg')
        # resize 1440*1024
        re_img = controller.img_resizing(bin_img)
        save_img = Image.fromarray(re_img)
        save_img.save(save_path+'3-resize.jpg')
        # Center Crop
        cropped_img = controller.center_crop(save_img)
        save_img = Image.fromarray(cropped_img)
        save_img.save(save_path+'4-center_crop.jpg')
        # bilateralFilter
        bi_img = controller.bilateral(cropped_img)
        save_img = Image.fromarray(bi_img)
        save_img.save(save_path+'5-filter.jpg')
        # dilation
        # di_img = controller.dilation(bi_img)
        di_img = controller.dilation2(bi_img)
        save_img = Image.fromarray(di_img)
        save_img.save(save_path+'6-dilation.jpg')
        # model - UNet
        seg_img = controller.unet_seg(di_img)
        save_img = Image.fromarray(seg_img)
        save_img.save(save_path+'7-seg_output.jpg')
        # contour detection
        contours, contour_cnt = controller.contour_detect(seg_img)
        if contour_cnt >= 3:
            inverted_img = controller.bg_img(di_img)
            save_img = Image.fromarray(inverted_img)
            save_img.save(save_path+'robo_'+img_name+'.jpg')
        else:
            seg_img = controller.unet_seg(bi_img)
            save_img = Image.fromarray(seg_img)
            save_img.save(save_path+'7-seg_output.jpg')
            contours, contour_cnt = controller.contour_detect(seg_img)

            inverted_img = controller.bg_img(bi_img)
            save_img = Image.fromarray(inverted_img)
            save_img.save(save_path+'robo_'+img_name+'.jpg')
        print('contour_cnt:', contour_cnt)
        # draw contour line
        draw_img = controller.draw_contour(seg_img, contours)
        save_img = Image.fromarray(draw_img)
        save_img.save(save_path+'8-contour.jpg')
        self.display_image(save_path+'8-contour.jpg')
        controller.generate_JSON(inverted_img, contours, img_name)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageUploader()
    window.show()
    sys.exit(app.exec_())