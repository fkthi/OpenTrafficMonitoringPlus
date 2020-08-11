import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import math
import random

COLORS = ((244, 67, 54), (233, 30, 99), (156, 39, 176), (103, 58, 183),
          (63, 81, 181), (33, 150, 243), (3, 169, 244), (0, 188, 212),
          (0, 150, 136), (76, 175, 80), (139, 195, 74), (205, 220, 57),
          (255, 235, 59), (255, 193, 7), (255, 152, 0), (255, 87, 34),
          (121, 85, 72), (158, 158, 158), (96, 125, 139))

color_map = {}


def get_color_mapping(id):
    if id in color_map:
        return color_map[id]
    color = random.choice(COLORS)
    color = (int(color[0]), int(color[1]), int(color[2]))
    color_map[id] = color
    return color


def roundup(x, fps):
    fps //= 4
    return int(math.ceil(x / float(fps))) * fps


def unit_mapping(val, meter_to_px):
    switch = {
        "speed": "km/h",
        "yaw": "rad",
        "yawInImg": "rad",
        "veh_length": "m",
        "veh_width": "m",
        "posX": "m",
        "posY": "m",
        "accx_LTP": "m/s",
        "accY_LTP": "m/s"
    }
    retval = switch.get(val, "")

    return retval if meter_to_px != 1 else retval.replace("km/h", "px/s")

def compress_video(video_path, abs_path, video_name):
    """Executes ffmpeg shell command to compress the output video"""
    print("Compressing video...")
    save_path = Path(abs_path).parents[0]
    save_path = str(save_path) + "/"+video_name +"_compressed.mp4"
    os.system("ffmpeg -i {} -vcodec libx265 -crf 28 {} -loglevel quiet -y".format(video_path, save_path))
    # Remove uncompressed Video
    os.remove(video_path)

def visualize_tracks(path, all_tracks, cfg):
    from postprocess import back_to_normal
    n_frames_smoothing = 20
    assert n_frames_smoothing % 2 ==0 ,"smoothing factor must be even"
    assert cfg.vis_mode in ["normal", "debug"], "Invalid Argument for Visualization mode"

    supported_metrics = [
        "speed", "yaw", "yawInImg", "posX", "posY", "veh_length", "veh_width",
        "accX_LTP", "accY_LTP"
    ]

    if not cfg.vis_metric in supported_metrics:
        print(
            "WARNING: desired metric '{}' is not supported, choose from {} \ndefaulting to 'speed' as metric "
            .format(cfg.vis_metric, supported_metrics))
        cfg.vis_metric = "speed"

    abs_path = os.path.abspath(path)
    video_name = Path(abs_path).parents[0].name
    filename = video_name.replace(" ", "_") + ".mp4"
    save_path = (Path(abs_path).parents[0]) / filename
    imgs_path = Path(abs_path).parents[0] / "temp/preprocessed_images"

    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'),
                             cfg.fps, (cfg.width, cfg.height))
    sorted_img_list = sorted(os.listdir(str(imgs_path)),
                             key=lambda x: int(x.split(".")[0]))

    for i, frame_name in enumerate(tqdm(sorted_img_list, disable=not cfg.verbose, desc="Creating visualization".center(30))):
        frame = cv2.imread(str(imgs_path / frame_name))
        tracks_in_current_frame = []
        for key, value in all_tracks.items():
            if i + 1 in value["corresponding_frame"]:
                tracks_in_current_frame.append(key)

        # get corresponding bb
        for track_id in tracks_in_current_frame:
            trk = all_tracks[track_id]
            # get position of bb

            idx_pos_of_bb = int(np.where(np.array(trk["corresponding_frame"]) == i + 1)[0])

            corresponding_bb = trk["carCorners_tracker"][idx_pos_of_bb]
            corresponding_bb = np.int0(corresponding_bb)

            if cfg.draw_kf_corners:
                corresponding_bb_kf = trk["carCornersKF_tracker"][idx_pos_of_bb]
                corresponding_bb_kf = back_to_normal(np.array(corresponding_bb_kf), cfg.meter_to_px,
                                                     cfg.orientation_offset, cfg.linear_offsets)
                corresponding_bb_kf = np.int0(corresponding_bb_kf)
                cv2.drawContours(frame, [corresponding_bb_kf], 0, (0, 0, 0), 2)

            try:
                roundup_idx = roundup(idx_pos_of_bb, cfg.fps)
                # Select range
                corresponding_metric = sum(
                    trk[cfg.vis_metric][roundup_idx - (n_frames_smoothing // 2):roundup_idx
                    +(n_frames_smoothing // 2)]) / n_frames_smoothing


            except:
                corresponding_metric = trk[cfg.vis_metric][-1]

            if cfg.vis_metric == "speed" and cfg.meter_to_px != 1:
                # Conversion from m/s -> km/h
                corresponding_metric *= 3.6

            if cfg.draw_corners:
                cv2.drawContours(frame, [corresponding_bb], 0, get_color_mapping(track_id), 4)
            x, y, w, h = cv2.boundingRect(corresponding_bb)

            text_metric = "{:4.1f}{} ID: {:2} ".format(corresponding_metric,
                                                      unit_mapping(cfg.vis_metric, cfg.meter_to_px), str(track_id))

            if cfg.vis_mode == "debug":
                font_size = 0.5
                line_type = 1
                box_margin = 3
                centroid = (np.sum(corresponding_bb[:,0]) // 4) - 25 , (np.sum(corresponding_bb[:,1]) // 4) + 25
                cv2.putText(frame, str(track_id), centroid, cv2.FONT_HERSHEY_SIMPLEX,1.4, (225, 0, 255), 3)

            elif cfg.vis_mode == "normal":
                font_size = 0.6
                line_type = 2
                box_margin = 10


            (text_width, text_height), baseline = cv2.getTextSize(text_metric, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_type)
            cv2.rectangle(frame, (x, y), (x + text_width, y - text_height - baseline - box_margin), get_color_mapping(track_id),
                          cv2.FILLED)
            cv2.putText(frame, text_metric, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), line_type)

        writer.write(frame)
    writer.release()
    if cfg.compress_video:
        compress_video(save_path, abs_path, video_name)
        filename = filename.replace(".mp4","") + "_compressed.mp4"
        save_path = str(save_path).replace(".mp4","") + "_compressed.mp4"
    print("\ncreated video '{}' in '{}'".format(filename, save_path))
