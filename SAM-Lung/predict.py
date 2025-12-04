import os
import time
import csv
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from sklearn.metrics import jaccard_score
import scipy.spatial
from scipy.spatial import distance_matrix
from medpy.metric.binary import hd95
import cv2
from base_model import BaseModel
# from base_model_sam import BaseModel


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def save_results_to_csv(file_name, data):
    file_exists = os.path.exists(file_name)
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['File Name', 'IoU', 'Dice Score', 'Pixel Accuracy', 'hd95'])
        writer.writerow(data)

def predict(img_path, roi_mask_path):
    print(img_path + " is begin!")
    classes = 1  # exclude background
    weights_path = r"multi_train/best_model.pth"
    mona_weights_path = "multi_train/mona_weights_best.pth"
    # img_path = "./DRIVE/test/images/02_test.tif"
    # roi_mask_path = "./DRIVE/test/mask/02_test_mask.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = BaseModel(n_channels=3, n_classes=classes+1, bilinear=True)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'], strict=False)
    model.image_encoder.load_mona_parameters(mona_weights_path)

    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = Image.fromarray(np.array(roi_img)).resize((1024, 1024), Image.NEAREST)
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    resize_transform = transforms.Resize((1024, 1024))

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([
                                         #resize_transform,
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)
                                         ])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval() 
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        prediction[prediction == 1] = 255
        opened = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8), iterations=5)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=5)

        prediction = closed
        mask = Image.fromarray(prediction)

        save_path = os.path.join('pspnet', img_path.split('/')[-4], img_path.split('/')[-3],
                                 img_path.split('/')[-1])

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        mask.save(save_path)

        target = (roi_img == 255).astype(np.uint8) 
        prediction = (prediction == 255).astype(np.uint8)

        pixel_accuracy = np.sum(target == prediction) / target.size

        if np.sum(target) == 0 and np.sum(prediction) == 0:
            iou = 1.0 
            dice_score = 1.0  
            hd95_score = 0
        else:
            iou = jaccard_score(target.flatten(), prediction.flatten(), average='binary')
            intersection = np.sum(target * prediction)
            dice_score = (2.0 * intersection) / (np.sum(target) + np.sum(prediction))
            if np.all(target == 255) and np.all(prediction == 255):
                hd95_score=0
            elif np.sum(target) == 0 or np.sum(prediction) == 0 :
                hd95_score = -1
            else :
                hd95_score = hd95(target, prediction)

        base_filename = os.path.basename(img_path)

        save_results_to_csv('output.tsv', [base_filename, iou, dice_score, pixel_accuracy, hd95_score])

def main():
    filelist="filelist.txt"
    with open(filelist, 'r') as file:
        for mask in file:
            mask = mask.strip()
            label = mask.replace('mask', 'label')
            predict(mask, label)


if __name__ == '__main__':
    main()
