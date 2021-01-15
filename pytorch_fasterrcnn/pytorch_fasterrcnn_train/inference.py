# This script does only inference from the loaded model
import cv2
import matplotlib.pyplot as plt
import torch
import model
import os
from PIL import Image
import torchvision.transforms as T
import config
import dataset
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torchvision.utils import save_image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

__all__ = [
    "load_model",
    "load_image_tensor",
    "get_prediction",
    "draw_box",
    "load_image_to_plot",
    "save_prediction",
    "get_folder_results",
]


def load_model():
    detector = model.create_model(num_classes=config.NUM_CLASSES , backbone=config.BACKBONE)
    # print(detector)

    detector.load_state_dict(torch.load(config.MODEL_SAVE_PATH , map_location=device)   )

    #print(detector)
    
    detector.eval()
    detector.to(device)
    return detector


# Load the detector for inference


def load_image_tensor(image_path, device):
    

    image_tensor = T.ToTensor()(Image.open(image_path))
    input_images = [image_tensor.to(device)]
    return input_images


def get_prediction(detector, images):
    # We can do a batch prediction as well but right now I'm doing on single image
    # Batch prediction can improve time but let's keep it simple for now.
    with torch.no_grad():
        prediction = detector(images)
        return prediction

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def draw_box(image, box, label_id, score):
    xtl = int(box[0])
    ytl = int(box[1])
    xbr = int(box[2])
    ybr = int(box[3])
    # Some hard coding for label
    if label_id == 1:
        label = "yes"
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr) ,  color=(0, 255, 0) )

    elif label_id == 2:
        label = "no"
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color=(0, 0, 255))
    elif label_id == 3:
        label = "invisible"
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color=(0, 0, 255))
    elif label_id == 4:
        label = "wrong"
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color=(0, 0, 255))
    elif label_id == 9:
        label = "groundtruth"
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color=(255, 0, 0))

    print("label = {}".format(label))
    cv2.putText(
        image, label, (xtl, ytl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2
    )
    # cv2.putText(image, label, (xbr, ybr), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (36,255,12), 2)


def load_image_to_plot(image_dir):
    image = cv2.imread(image_dir, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_prediction(prediction, image_name, image , indexi):
   #trying to extract the box with the bboxes
    train_df = pd.read_csv(config.TRAIN_CSV_PATH)

    train_dataset = dataset.detection_dataset(
        train_df,
        config.IMAGE_DIR,
        target=config.TARGET_COL,
        train=True,
        transforms=T.Compose([T.ToTensor()]),
    )
    

    #PREDICTION PART 
    for pred in prediction:
        boxes = pred["boxes"].data.cpu().numpy()
        labels = pred["labels"].data.cpu().numpy()
        scores = pred["scores"].data.cpu().numpy()

    image_id = train_dataset.image_ids[indexi]
    print(image_id,"img_id")
           
    records = train_dataset.df[train_dataset.df["image_id"] == image_id]
    gtboxes = records[["xtl", "ytl", "xbr", "ybr"]].values
    gtboxes = torch.as_tensor(gtboxes, dtype=torch.int)
    gtboxes = torch.reshape(gtboxes, (-1,))
    print(gtboxes , "gtboxes")

    

    draw_box(image, gtboxes, 9 , 0)

    for i in range(len(labels)) :
        if scores[i] > config.DETECTION_THRESHOLD:
            box_draw = boxes[i]
            print(boxes , "boxes")

            label_draw = labels[i]
            print(label_draw , "LABEEEELLLLLSSSSSSS")

            score = scores[i]
                    
            IoU = bb_intersection_over_union(box_draw, gtboxes)     
            print(IoU , "IIIIIIIIoooooooooUUUUUUUUUUUU")
                    
            print(score,"score")
            print(box_draw,"box_draw")
            print(label_draw,"label_draw")
            draw_box(image, box_draw, label_draw, score)
            

    #save_image(image, '%sprediction_%03d.jpg' %
    #        (config.SAVE_DIR, indexi))

    #plt.imshow(image)
    #plt.show()

    image_name = config.OUTPUT_PATH + image_name
    cv2.imwrite(image_name, image)


def get_folder_results(detector, image_dir, device):
    i=0 
    for image  in os.listdir(image_dir) :
        image_path = os.path.join(image_dir, image)
        if image_path == image_dir + '.DS_Store':
            continue
        input_images = load_image_tensor(image_path, device)
        prediction = get_prediction(detector, input_images)
        print(prediction , "    prediction")
        image_loaded = load_image_to_plot(image_path)
        save_path = os.path.join(config.SAVE_DIR, image)
        save_prediction(prediction, save_path, image_loaded , i )
        i=i+1


if __name__ == "__main__":

    detector = load_model()
    print("---------- Model succesfully loaded -------- ")
    # print(detector)

    input_images = load_image_tensor(config.PREDICT_IMAGE, device)
    prediction = get_prediction(detector, input_images)
    # print(prediction)
    image = load_image_to_plot(config.PREDICT_IMAGE)
    #save_prediction(prediction, config.SAVE_IMAGE, image)
    get_folder_results(detector, config.IMAGE_DIR, device)
    # print(prediction)
