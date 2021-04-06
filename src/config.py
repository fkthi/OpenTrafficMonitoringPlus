import numpy as np


class Config():
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)


default_config = Config({
    "config_name": "default",
    # If 'video_name' is in the actual videos name -> this is the corresponding config
    "video_name": None,
    # Limits console output to a minimum if set to False
    "verbose": True,

    # <-------- PREPROCESSING -------->

    # limits the number of frames. set to -1 for entire video
    "num_frames": -1,
    # Processes (CPU cores) used for image registration
    "num_workers": 4,
    # Whether to register the images (fixed frame for all images - important for tracking!)
    "register_images": False,
    # Keypoint Detection algorithm used for registration. Either "surf" or "orb"
    "kp_algo": "surf",
    # hessian_threshold value for keypoint computation.
    # Lower Value -> better results and vice versa.
    # We achieved good results with values from 2000 to 5000
    "hessian_threshold": 5000,
    # Quality of extracted and preprocessed images. Ranging from 0 to 100
    # The quality effects jpg file size and the overall processing time
    "jpg_quality": 95,
    # Required for Images -> has no effect when using videos,
    # since fps information is obtained from video file
    "fps": 50,

    # <-------- MASKRCNN -------->

    # Limits the number of detections per image. If you run out of GPU memory during inference,
    # reducing this value is a good idea.
    "detections_per_image": 150,
    # Weights used for inference
    "weights_path": "maskrcnn/model_final.pth",
    # Detectron2 config used for inference. (Has nothing to do with this config)
    "detectron2_config": "./maskrcnn/mask_rcnn_R_50_FPN_1x.yaml",
    # Number of classes in the model
    "num_classes": 1,
    # Save predicted Images from MASKRCNN in 'temp/images_after_network'
    # -> reduces performance but good for quality evaluation
    "save_predicted": False,
    # Score threshold for the detections
    "score_threshold": 0.5,

    # <-------- TRACKER -------->

    # If the speed of a vehicle is less than 'standstill'
    # it's considered to be not moving (in m/s)
    "standstill": 0.5,
    # The threshold value for which a vehicle is considered to have
    # "fully" entered the frame (vehicle length in %)
    # Only after the vehicle has "fully" entered the Frame it gets tracked
    "fully_entered_threshold": 80,
    # If the average speed of a track / object is less than 'standstill' the track gets discarded
    "discard_slow_tracks": False,
    # If the track is lost for n consecutive frames, it is considered to be lost
    "invisible_for_too_long": 5,
    # Hungarian algorithm cost of non assignment
    "cost_of_non_assignment": 0.499,
    # Minimal number of frames per track. if less than min_hits the track gets discarded
    "min_hits": 10,

    # <-------- RESULTS -------->

    # Spatial resolution / Ground Sampling Distance: meter to pixel conversion value.
    # For results in pixel set the value to 1
    "meter_to_px": 0.035,
    # translational offset, used in the publications for benchmark with reference sensor
    "linear_offsets": np.array([[0], [0]]),  # [x, y] in meters
    # orientational offset:
    # to obtain same orientations, e.g. standardized orientation according to
    # Ground Control Points (used for the publications)
    "orientation_offset": 0,  # in radians
    # Drone flight height in meters
    "drone_height": 50,
    # to reduce effect of the relief displacement:
    # average height of the outer vehicle part (vehicle shoulder)
    "outer_height": 0.7,
    # to reduce effect of the relief displacement:
    # average height of the inner vehicle part (vehicle ground clearance)
    "inner_height": 0.15,
    # Choose correction for relief displacement:
    # 0 --> no correction (position equals centroid of detection)
    # 1 --> correction according to "outer_height" and "inner_height"
    # 2 --> vehicle size known (publications)
    "radial_displacement": 0,
    # Output format for finished tracks
    "save_as_csv": True,
    "save_as_json": True,
    "save_as_pickle": True,
    "save_as_txt": True,
    # Whether to delete the temporal images after the video is finished
    "del_temp_images": True,

    # <-------- VISUALIZATION  -------->

    # Create track visualization video after postprocessing
    "visualize": True,
    # Draw detected car corners in the visualization video
    "draw_corners": True,
    # Draw kalman car corners in the visualization video
    "draw_kf_corners": False,
    # Which metric to use for the visualization video.
    # For a list of the supported metrics go to visualize_tracks.py
    "vis_metric": "speed",
    # Choose between 'debug' or 'normal'
    "vis_mode": "normal",
    # Compress Video with ffmpeg (requires ffmpeg to be installed)
    "compress_video": False
})

example_new_config = default_config.copy({
    "config_name": "test_config",
    "video_name": "traffic",
    "verbose": True,
    "min_hits": 20

    # ....
})

configs_to_consider = [example_new_config]


def config_by_config_name(config_name):
    if config_name == "default":
        return config_default()

    for config in configs_to_consider:
        if config.config_name == config_name:
            return config
    raise ValueError("Config not found ")


def config_default():
    return default_config


def config_by_video_name(video_name):
    for config in configs_to_consider:
        if config.video_name in video_name:
            print("\nUsing config : '{}' for Video: '{}'".format(
                config.config_name, video_name))
            return config
    print(
        "WARNING: no specific config found for '{}'. Make sure to create a"
        " custom config and add it to the 'configs_to_consider' list in 'src/config.py'."
        "\nIf you have modified the default config you can ignore this warning\n"
            .format(video_name))
    return default_config
