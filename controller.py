import os
import cv2
import json
import numpy as np
from PIL import Image

import torch
from model.unet_single_inference import UNetSingleInference

def img_path(img_path):
    img_filename = os.path.basename(img_path).replace('.jpg', '').replace('.png', '')
    save_path = './result/'+img_filename+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return img_filename, save_path

def img_binary(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def img_resizing(img):
    pil_image = Image.fromarray(img)
    resized_pil_image = pil_image.resize((1440, 1024), Image.LANCZOS)
    return np.array(resized_pil_image)

def center_crop(img):
    width, height = img.size
    if width < 1440 or height < 1024:
        img = img.resize((1440, 1024), Image.LANCZOS)
        width, height = img.size
    
    new_size = 512
    left = int((width - new_size) / 2)
    top = int((height - new_size) / 2)
    right = int(left + new_size)
    bottom = int(top + new_size)

    cropped = img.crop((left, top, right, bottom))
    return np.asarray(cropped)

def bilateral(img):
    return cv2.bilateralFilter(img, -1, 10, 10)

def dilation(img):
    kernel_size = (3,3)
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(img, kernel, 1)

def dilation2(img):
    inverted_img = cv2.bitwise_not(img)
    kernel_size = (3,3)
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(inverted_img, kernel, iterations=1)
    result_img = cv2.bitwise_not(dilated)
    return result_img

def unet_seg(img):
    ckpt_dir = './ckpt/model_epoch50.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if_manager = UNetSingleInference(model_path=ckpt_dir, device=device)
    output = if_manager.inference(img)
    output_img = (output[0, :, :, 0] * 255).astype(np.uint8)
    return output_img

def contour_detect(img):
    draw_img = img.copy()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
    num_contours = len(contours)
    return contours, num_contours

def bg_img(img):
    pro_di_img = (img[ :, :] * 255).astype(np.uint8)
    inverted_img = cv2.bitwise_not(pro_di_img)
    return inverted_img

def draw_contour(img, contours):
    draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.drawContours(draw_img, contours, -1, (0, 255, 0), 2)

def generate_JSON(img, contours, img_filename):
    image_id = 1
    category_id = 1

    annotations = []
    annotation_id = 1
    print(img_filename)
    images = [{
        "id": image_id,
        "file_name": 'robo_'+img_filename+'.jpg',
        "height": 512,
        "width": 521
    }]

    categories = [{
        "id": category_id,
        "name": "object",
        "supercategory": "none"
    }]

    for contour in contours:
        seg_ = contour.flatten().tolist()
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [x, y, w, h]
        area = cv2.contourArea(contour)
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [seg_],
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        }
        annotations.append(annotation)
        annotation_id += 1
    
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    json_path = './result/'+img_filename+'/robo_'+img_filename+'.json'
    with open(json_path, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)