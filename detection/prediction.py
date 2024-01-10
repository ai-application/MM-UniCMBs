from ultralytics import YOLO
import os
import cv2
import numpy as np
import math
import tqdm
import copy

imgs_path = "test_images"
out_path = "test_results"

crop_image_path = 'cmb_results'

os.makedirs(out_path, exist_ok=True)

# Load a model
model = YOLO("CMB.pt")  # load a custom model

# Predict with the model
img_files = os.listdir(imgs_path)

for file_index in tqdm.tqdm(range(len(img_files))):
    file = img_files[file_index]

    base_name = file[:-4]

    cv_img = cv2.imdecode(np.fromfile(os.path.join(imgs_path, file), dtype=np.uint8), -1)

    src_img = copy.deepcopy(cv_img)

    h, w, c = cv_img.shape

    results = model.predict(cv_img, imgsz = w, augment = True)  # predict on an image

    detect_boxes = results[0].boxes.cpu().numpy()
    max_length = 0.0
    max_yhh_box = None

    have_moss = False

    for box in detect_boxes:
        xmin, ymin, xmax, ymax = [int(val) for val in box.xyxy.squeeze()]

        score, cls = box.conf.squeeze(), box.cls.squeeze()

        length = int(math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2))

        if score >= 0.5:

            #图像得2/3处以下才是感兴趣得区域
            #if (ymin + ymax) // 2 >= 750:
            if True:
                cv2.rectangle(cv_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

                cv2.putText(cv_img, str(score), (xmin - 15, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imencode('.jpg', cv2.hconcat([src_img, cv_img]))[1].tofile(os.path.join(out_path, base_name + "result.jpg"))

    cv2.imencode('.jpg', cv_img)[1].tofile(os.path.join(out_path, base_name + "result.jpg"))