import cv2
import numpy as np
from tqdm import trange, tqdm
import os
import sys
from multiprocessing import Pool
import functools
import time
import shutil


def _register_image(img_to_align_name, ref_image, raw_img_folder, temp_folder, hessian_threshold, jpg_quality, kp_algo):
    image_idx = img_to_align_name
    img_to_align_name = cv2.imread(raw_img_folder + img_to_align_name)
    align = cv2.cvtColor(img_to_align_name, cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    height, width = ref.shape

    if kp_algo.lower() == "surf":
        kp = cv2.xfeatures2d.SURF_create(hessian_threshold)
        norm = cv2.NORM_L1
        x_check = False
    else:
        kp = cv2.ORB_create()
        norm = cv2.NORM_HAMMING
        x_check = True

    if "ref_kp" not in _register_image.__dict__:
        _register_image.ref_kp, _register_image.ref_d = kp.detectAndCompute(ref, None)


    kp1, d1 = kp.detectAndCompute(align, None)
    matcher = cv2.BFMatcher(norm, crossCheck=x_check)

    matches = matcher.match(d1, _register_image.ref_d)
    matches.sort(key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = _register_image.ref_kp[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    cv2.imwrite(temp_folder + "/preprocessed_images/" + image_idx,
                cv2.warpPerspective(img_to_align_name, homography, (width, height)),
                [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])


def register_images(temp_dir, temp_folder_raw, temp_folder_reg, ref_img, cfg):
    assert cfg.kp_algo.lower() in ["surf", "orb"], "Keypoint algorithm '{}' not supported" .format(cfg.kp_algo)
    p_bar = tqdm(total=len(os.listdir(temp_folder_raw)), disable=not cfg.verbose, desc="Register frames".center(30))
    p = Pool(cfg.num_workers).map_async(functools.partial(_register_image, ref_image=ref_img,
                                                          raw_img_folder=temp_folder_raw, temp_folder=temp_dir,
                                                          hessian_threshold=cfg.hessian_threshold,
                                                          jpg_quality=cfg.jpg_quality, kp_algo=cfg.kp_algo),
                                        os.listdir(temp_folder_raw))
    while True:
        a = len(os.listdir(temp_folder_raw))
        b = len(os.listdir(temp_folder_reg))
        p_bar.n = (a - (a - b))
        p_bar.refresh()
        if a == b:
            break
        time.sleep(1)
    p_bar.close()
    p.wait()

    [os.remove(temp_folder_raw + file) for file in os.listdir(temp_folder_raw)]

def sort_image_names(img_names):
    # Check if image names are just numbers
    if all([img.split(".")[0].isnumeric() for img in img_names]):
        print("Sorting Images numerically")
        return sorted(img_names, key= lambda x : int(x.split(".")[0]))

    # Check if in Format video_xxx_001, video_xxx_002, ...
    if all(["_" in img for img in img_names]) and all([img.split(".")[0].split("_")[-1].isnumeric()
                                                       for img in img_names]):

        print("Sorting Images by number after last underscore")
        return sorted(img_names, key=lambda x : int(x.split(".")[0].split("_")[-1]))
    # Default
    print("WARNING: No Image naming pattern detected. Defaulting to alphabetic order")
    return sorted(img_names)


def preprocess_images(image_folder, temp_folder, cfg):
    assert len([el for el in os.listdir(image_folder) if el.endswith(".jpg")]) > 0, "No .jpg files found in Folder: " \
                                                                                    + image_folder
    if cfg.register_images:
        raw_img_folder = os.path.join(temp_folder, "raw_images/")
        reg_img_folder = os.path.join(temp_folder, "preprocessed_images/")
        if not os.path.exists(raw_img_folder):
            os.mkdir(raw_img_folder)
        if not os.path.exists(reg_img_folder):
            os.mkdir(reg_img_folder)

        for image in os.listdir(image_folder):
            shutil.copyfile(os.path.join(image_folder, image), os.path.join(raw_img_folder, image))

        sorted_img_list = sort_image_names(os.listdir(raw_img_folder))
        for i, img in enumerate(sorted_img_list):
            os.rename(os.path.join(raw_img_folder, img), os.path.join(raw_img_folder, str(i) + ".jpg"))

        ref_img = cv2.imread(os.path.join(raw_img_folder, "0.jpg"))

        cfg.height = ref_img.shape[0]
        cfg.width = ref_img.shape[1]

        register_images(temp_folder, raw_img_folder, reg_img_folder, ref_img, cfg)
    else:
        reg_img_folder = os.path.join(temp_folder, "preprocessed_images/")
        if not os.path.exists(reg_img_folder):
            os.mkdir(reg_img_folder)

        for image in os.listdir(image_folder):
            shutil.copyfile(os.path.join(image_folder, image), os.path.join(reg_img_folder, image))
        sorted_img_list = sort_image_names(os.listdir(reg_img_folder))
        for i, img in enumerate(sorted_img_list):
            os.rename(os.path.join(reg_img_folder, img), os.path.join(reg_img_folder, str(i) + ".jpg"))
        ref_img = cv2.imread(image_folder + sorted_img_list[0])

        cfg.height = ref_img.shape[0]
        cfg.width = ref_img.shape[1]

def _convert_image_name(img_name, total_frames):
    l1 = len(str(img_name))
    l2 = len(str(total_frames))
    if l1 < l2:
        # Add leading zeros if necessary
        img_name = ("0" * (l2 - l1)) + str(img_name)
    return str(img_name)

def preprocess(video_path, temp_dir, cfg):
    """
    Performs preprocessing on a video
        1. breaks down the video into raw frames
        2. registers all images according to the first

    num workers -> how many parallel processes
    """
    print("Processing Video '{}' with Mask RCNN weights: {}\n".format(temp_dir.split("/")[2],cfg.weights_path), file=sys.__stdout__)

    if cfg.verbose:
        sys.stdout = sys.__stdout__
    else:
        sys.stdout = open(os.devnull, 'w')

    cap = cv2.VideoCapture(video_path)

    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    success, image = cap.read()

    cfg.height = image.shape[0]
    cfg.width = image.shape[1]
    cfg.fps = frame_rate

    if cfg.num_frames == -1:
        cfg.num_frames = float("inf")

    assert success, "video capture not working properly"
    ref_img = image
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    temp_folder_reg = temp_dir + "preprocessed_images/"
    temp_folder_raw = temp_dir + "raw_images/"

    if not os.path.exists(temp_folder_reg):
        os.mkdir(temp_folder_reg)
    if not os.path.exists(temp_folder_raw):
        os.mkdir(temp_folder_raw)

    if len(os.listdir(temp_folder_reg)) == min(cfg.num_frames, total_frame -1):
        print("found existing preprocessed images in {} -> skipping preprocessing".format(temp_folder_reg))
        return

    # Check if frames have already been extracted
    if len(os.listdir(temp_folder_raw)) == min(cfg.num_frames, total_frame -1):
        print("found extracted images in {} -> skipping extraction".format(temp_folder_raw))

    else:
        extraction_folder = temp_folder_raw if cfg.register_images else temp_folder_reg

        for i in trange(min(cfg.num_frames, total_frame), disable=not cfg.verbose, desc="Extracting video to frames".center(30)):
            if success and i < cfg.num_frames:
                if not (image.shape[0] == cfg.height and image.shape[1] == cfg.width):
                    image = cv2.resize(image, (cfg.width, cfg.height))
                cv2.imwrite(extraction_folder + _convert_image_name(i +1, total_frame) + ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpg_quality])
                success, image = cap.read()

    if cfg.register_images:
        register_images(temp_dir, temp_folder_raw, temp_folder_reg, ref_img, cfg)
        return
    print("'register_image' is set to False -> skipping registration")
