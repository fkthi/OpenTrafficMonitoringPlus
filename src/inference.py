from pathlib import Path
import cv2
import os
import numpy as np
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.utils import visualizer
from detectron2.engine.defaults import DefaultPredictor

def evalimage_get_contour(model, path, save_predicted):
    image = cv2.imread(path)
    pred = model(image)['instances']

    if save_predicted:
        image_name = path.split("/")[-1]
        v = visualizer.Visualizer(image, {})
        image_object = v.draw_instance_predictions(pred.to("cpu"))
        save_path = str(Path(os.path.abspath(path)).parents[1]) \
                    + "/images_after_network/" + image_name
        cv2.imwrite(save_path, image_object.get_image())

    masks = pred.pred_masks.cpu().numpy()
    box_img = []

    for i_mask in range(len(masks)):
        if np.count_nonzero(masks[i_mask]) == 0:
            # Mask is zero
            continue
        mask = masks[i_mask].astype('uint8') * 255
        _, contour, _ = cv2.findContours(mask, 1, 1)

        # default contour index (if only 1 contour)
        contour_idx = 0
        contour = np.array(contour)

        if contour.ndim == 1:
            contour_shape = contour.shape
            # number of contours found
            num_contours = contour_shape[0]
            area_cnt = np.ones(num_contours, )
            for cnt_idx in range(0, num_contours):
                area_cnt[cnt_idx] = cv2.contourArea(contour[cnt_idx])
            # overwrite contour_idx according to the largest contour found
            contour_idx = np.argmax(area_cnt)

        cnt = contour[contour_idx]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box_img.append(box)

    return box_img


def run_inference(temp_folder, opentraffic_config):
    if os.path.exists(os.path.join(temp_folder, "data/boxes.npy")):
        print("found existing Mask RCNN Predictions in {}"
              " -> skipping Prediction".format(temp_folder + "data/"))
        return

    cfg = get_cfg()
    cfg.merge_from_file(opentraffic_config.detectron2_config)
    cfg.INPUT.MAX_SIZE_TEST = 1920
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.TEST.DETECTIONS_PER_IMAGE = opentraffic_config.detections_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = opentraffic_config.weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = opentraffic_config.score_threshold

    num_images = len([
        name for name in os.listdir(temp_folder + "preprocessed_images")
        if ".jpg" in name
    ])
    boxes_rot_cell = np.zeros((num_images), dtype=np.object)

    model = DefaultPredictor(cfg)

    img_path = temp_folder + "/preprocessed_images/"
    data_path = temp_folder + "/data/"

    if opentraffic_config.save_predicted:
        if not os.path.exists(temp_folder + "images_after_network/"):
            os.mkdir(temp_folder + "images_after_network/")

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    sort_img_path = sorted(os.listdir(img_path),
                           key=lambda x: int(x.split(".")[0]))

    desc = "Predicting Masks and Images" if opentraffic_config.save_predicted else "Predicting Masks"

    for i, image_name in enumerate(tqdm(sort_img_path, desc=desc.center(30))):
        boxes_rot_cell[i] = evalimage_get_contour(
            model, img_path + image_name, opentraffic_config.save_predicted)

    np.save(temp_folder + "data/boxes.npy", boxes_rot_cell)
