import warnings
import csv
import os
import sys
import math
from operator import itemgetter
from pathlib import Path

warnings.simplefilter("ignore", RuntimeWarning)

from scipy.optimize import linear_sum_assignment
from tqdm import trange
import numpy as np
from shapely.geometry import Polygon
from visualize_results import visualize_tracks


class TrackEnum():
    """
    Enum for safer use
    """
    ID = "id"
    KALMANFILTER = "kalman_filter"
    AGE = "age"
    TOTALVISIBLECNT = "total_visible_count"
    CONSECINVISCNT = "consecutive_invisible_count"
    TRACKID = "track_id"


class Tracks():
    """
    struct that contains all the current active tracks and manages their creation and deletion
    """

    def __init__(self):
        self.data = []
        self.next_id = 0

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_tracks(self):
        return self.data

    def create_entry_with_kf(self, kf_):
        self.data.append({TrackEnum.ID: self.next_id, TrackEnum.KALMANFILTER: kf_, TrackEnum.AGE: 1,
                          TrackEnum.TOTALVISIBLECNT: 1, TrackEnum.CONSECINVISCNT: 0})
        self.next_id += 1

    def delete_tracks(self, idx_list):
        if len(idx_list) == 0:
            return
        for del_idx in idx_list:
            if self.data[del_idx][TrackEnum.TOTALVISIBLECNT] >= cfg.min_hits:
                process_finished_track(self.data[del_idx])
            self.data[del_idx] = None
        for _ in range(len(idx_list)):
            self.data.remove(None)


def vehicle_resize_l_shape(car_corners, picture_centroids, drone_height,
                           meter_to_px):
    # Distance to Image Center
    dist_to_center = np.zeros((4, 14))
    # calculating the distance from all points of the
    # polygon to the center of the image
    for i in range(4):
        # Change in the x axis
        dist_to_center[i, 0] = car_corners[i, 0] - picture_centroids[-1, 0]
        # Change in the y axis
        dist_to_center[i, 1] = car_corners[i, 1] - picture_centroids[-1, 1]
        # Euclidean distance from image center to each corner
        dist_to_center[i, 2] = math.sqrt(dist_to_center[i, 0]
                                         ** 2 + dist_to_center[i, 1] ** 2)
        # Save corner label
        dist_to_center[i, 3] = i
        # Saving corner x coordinate
        dist_to_center[i, 4] = car_corners[i, 0]
        # Saving corner y coordinate
        dist_to_center[i, 5] = car_corners[i, 1]
    # Ordering the corners according to the center of the picture
    dist_to_center = np.array(sorted(dist_to_center, key=itemgetter(2)))

    for i in range(1, 4):
        dist_to_center[i, 6] = math.sqrt((dist_to_center[i, 4] - dist_to_center[0, 4])
                                         ** 2 + (dist_to_center[i, 5] - dist_to_center[0, 5]) ** 2)

    dist_to_center = np.array(sorted(dist_to_center, key=itemgetter(6)))
    # Drone height in pixels
    drone_height_in_px = drone_height / meter_to_px
    # vehicle ground clearance and height of the vehicle shoulder
    vehicle_inner_heights = [0.15, 0.75]
    inner_handover_radius_in_px = (drone_height * (0.061889 / 0.65)) / meter_to_px

    # Determining the inner corners
    if dist_to_center[0, 2] > inner_handover_radius_in_px:
        dist_to_center[0, 7] = vehicle_inner_heights[0]
    else:
        dist_to_center[0, 7] = vehicle_inner_heights[1]

    # computing correctION in X axis
    dist_to_center[0, 8] = (dist_to_center[0, 7] * dist_to_center[0, 0]) / drone_height_in_px
    # computing correctION in Y axis
    dist_to_center[0, 9] = (dist_to_center[0, 7] * dist_to_center[0, 1]) / drone_height_in_px
    # Adding correction to distance to center in X axis
    dist_to_center[0, 10] = dist_to_center[0, 0] - dist_to_center[0, 8]
    # Adding correction to distance to center in Y axis
    dist_to_center[0, 11] = dist_to_center[0, 1] - dist_to_center[0, 9]
    # Computing corrected coordiante in X axis
    dist_to_center[0, 12] = dist_to_center[0, 10] + picture_centroids[0, 0]
    # Computing corrected coordiante in Y axis
    dist_to_center[0, 13] = dist_to_center[0, 11] + picture_centroids[1, 1]

    # vehicle with in pixels
    vehicle_width_in_px = 1.842 / meter_to_px
    # rescale factor of the width
    dist_to_center[1, 7] = vehicle_width_in_px / dist_to_center[1, 6]
    # vehicle length in pixels
    vehicle_length_in_px = 4.725 / meter_to_px
    # rescale factor of the length
    dist_to_center[2, 7] = vehicle_length_in_px / dist_to_center[2, 6]
    dist_to_center[3, 7] = vehicle_length_in_px / dist_to_center[2, 6]
    # scaling car with known size
    for i in range(1, 4):
        # computing corrected size in pixels in X axis
        dist_to_center[i, 8] = (dist_to_center[i, 0] - dist_to_center[0, 0]) * \
                               dist_to_center[i, 7]
        # computing corrected size in pixels in Y axis
        dist_to_center[i, 9] = (dist_to_center[i, 1] - dist_to_center[0, 1]) * \
                               dist_to_center[i, 7]
        # Adding correction to distance to center in X axis
        dist_to_center[i, 10] = dist_to_center[i, 8] + dist_to_center[0, 0]
        # Adding correction to distance to center in Y axis
        dist_to_center[i, 11] = dist_to_center[i, 9] + dist_to_center[0, 1]
        # Computing corrected coordinate in X axis
        dist_to_center[i, 12] = dist_to_center[i, 10] + picture_centroids[0, 0]
        # Computing corrected coordinate in Y axis
        dist_to_center[i, 13] = dist_to_center[i, 11] + picture_centroids[1, 1]

    dist_to_center = np.array(sorted(dist_to_center, key=itemgetter(3)))
    car_corners[0:4, 0:2] = dist_to_center[0:4, 12:14]
    return car_corners


def vehicle_resize_polygon(car_corners, picture_centroids, drone_height, meter_to_px, outer_height,
                           inner_height):
    dist_to_center = np.zeros((4, 13))
    # calculating the distance from all points of the polygon to the center of the image
    for i in range(4):
        # Change in the x axis
        dist_to_center[i, 0] = car_corners[i, 0] - picture_centroids[0, 0]
        # Change in the y axis
        dist_to_center[i, 1] = car_corners[i, 1] - picture_centroids[1, 1]
        # Euclidean distance from image center to each corner
        dist_to_center[i, 2] = math.sqrt(dist_to_center[i, 0]
                                         ** 2 + dist_to_center[i, 1] ** 2)
        # Save corner label
        dist_to_center[i, 3] = i
        # Saving corner x coordinate
        dist_to_center[i, 4] = car_corners[i, 0]
        # Saving corner y coordinate
        dist_to_center[i, 5] = car_corners[i, 1]
    # Ordering the corners according to the center of the picture
    dist_to_center = np.array(sorted(dist_to_center, key=itemgetter(2)))

    for i in range(1, 4):
        dist_to_center[i, 6] = math.sqrt((dist_to_center[i, 4] - dist_to_center[0, 4])
                                         ** 2 + (dist_to_center[i, 5] - dist_to_center[0, 5]) ** 2)

    dist_to_center = np.array(sorted(dist_to_center, key=itemgetter(6)))
    # Drone height in pixels
    drone_height_in_px = drone_height / meter_to_px

    outer_height_in_px = outer_height / meter_to_px
    inner_height_in_px = inner_height / meter_to_px

    # The Side closest to the center is scaled with inner height
    for i in [0, 2]:
        # computing correction in X axis
        dist_to_center[i, 7] = (inner_height_in_px * dist_to_center[i, 0]) / drone_height_in_px
        # computing correction in Y axis
        dist_to_center[i, 8] = (inner_height_in_px * dist_to_center[i, 1]) / drone_height_in_px
        # Adding correction to distance to center in X axis
        dist_to_center[i, 9] = dist_to_center[i, 0] - dist_to_center[i, 7]
        # Adding correction to distance to center in Y axis
        dist_to_center[i, 10] = dist_to_center[i, 1] - dist_to_center[i, 8]
        # computing corrected coordinate in X axis
        dist_to_center[i, 11] = dist_to_center[i, 9] + picture_centroids[0, 0]
        # computing corrected coordinate in X axis
        dist_to_center[i, 12] = dist_to_center[i, 10] + picture_centroids[1, 1]

    for i in [1, 3]:
        # computing correction in X axis
        dist_to_center[i, 7] = (outer_height_in_px * dist_to_center[i, 0]) / drone_height_in_px
        # computing correction in Y axis
        dist_to_center[i, 8] = (outer_height_in_px * dist_to_center[i, 1]) / drone_height_in_px
        # Adding correction to distance to center in X axis
        dist_to_center[i, 9] = dist_to_center[i, 0] - dist_to_center[i, 7]
        # Adding correction to distance to center in Y axis
        dist_to_center[i, 10] = dist_to_center[i, 1] - dist_to_center[i, 8]
        # computing corrected coordinates in X axis
        dist_to_center[i, 11] = dist_to_center[i, 9] + picture_centroids[0, 0]
        # computing corrected coordinates in Y axis
        dist_to_center[i, 12] = dist_to_center[i, 10] + picture_centroids[1, 1]

    dist_to_center = np.array(sorted(dist_to_center, key=itemgetter(3)))
    car_corners[0:4, 0:2] = dist_to_center[0:4, 11:14]
    return car_corners


def length_class(car_corners, resolution, meter_to_px, orientation_offset, linear_offsets,
                 drone_height, outer_height, inner_height, radial_displacement):
    """
    Computes the final measurement vector for the kalman filter which
     contains the processed centroid of the vehicle and its orientation
    Also returns the processed car corners and the length, width.... etc
    """
    measurement_vec = [0, 0, 0]
    car_corners = np.matrix(car_corners)
    # Picture Centroids
    pic_centroids = np.array([[resolution[0] / 2, resolution[1]],
                              [resolution[0], resolution[1] / 2],
                              [resolution[0] / 2, 0],
                              [0, resolution[1] / 2],
                              [resolution[0] / 2, resolution[1] / 2]],
                             dtype=np.object)

    if radial_displacement == 2:
        car_corners = vehicle_resize_l_shape(car_corners, pic_centroids, drone_height,
                                             meter_to_px)
    elif radial_displacement == 1:
        car_corners = vehicle_resize_polygon(car_corners, pic_centroids, drone_height,
                                             meter_to_px, outer_height, inner_height)

    pic_centroids = pic_centroids * meter_to_px
    # Rotation matrix of 180 deg around x axis
    r_x = np.matrix([[1, 0], [0, -1]])
    # Rotating the picture centroids 180 deg around x axis
    for rotate_gcp_idx in range(len(pic_centroids)):
        pic_centroids[rotate_gcp_idx, 0:2] = (
                r_x.T @ pic_centroids[rotate_gcp_idx, 0:2].T).T.reshape(2, )
    # Rotation matrix to compensate the orientation offset
    rotation_matrix = np.matrix(
        [[math.cos(orientation_offset), -math.sin(orientation_offset)],
         [math.sin(orientation_offset), math.cos(orientation_offset)]])

    # Rotating the picture centroids to compensate the orientation offset
    for picture_frame_index in range(len(pic_centroids)):
        pic_centroids[picture_frame_index, 0:2] = rotation_matrix.T \
                                                  @ pic_centroids[picture_frame_index,
                                                    0:2].T.reshape(2, )

    pic_centroids[:, 0:2] = pic_centroids[:, 0:2] - linear_offsets.T
    # Mapping the corners of the bounding box from picture
    # coordinate frame to local tangent Plane
    car_corners = np.c_[car_corners, [1, 2, 3, 4]]
    # Applying the spatial resolution to the bounding box
    car_corners[:, 0:2] = car_corners[:, 0:2] * meter_to_px

    # Rotating the corners of the bounding box 180 deg around x axis
    for rotate_gcp_idx in range(len(car_corners)):
        car_corners[rotate_gcp_idx, 0:2] = (r_x.T @ car_corners[rotate_gcp_idx, 0:2].T).T

    # Rotating the corners of the bounding box to compensate the orientation offset
    for rotate_gcp_idx in range(len(car_corners)):
        car_corners[rotate_gcp_idx, 0:2] = (
                rotation_matrix.T * car_corners[rotate_gcp_idx, 0:2].T).T

    car_corners[:, 0:2] = car_corners[:, 0:2] - linear_offsets.T
    # Identifying the inner corners of the bounding box
    distance_to_img_center = np.zeros((4, 5))
    for k in range(4):
        distance_to_img_center[k, 0] = math.sqrt(((pic_centroids[4, 0] - car_corners[k, 0]) ** 2)
                                                 + (pic_centroids[4, 1] - car_corners[k, 1]) ** 2)

        distance_to_img_center[k, 1] = k
    distance_to_img_center = np.array(sorted(distance_to_img_center, key=itemgetter(0)))

    for k in range(4):
        distance_to_img_center[k, 2] = math.sqrt(
            (car_corners[int(distance_to_img_center[0, 1]), 0] -
             car_corners[int(distance_to_img_center[k, 1]), 0]) ** 2 +
            (car_corners[int(distance_to_img_center[0, 1]), 1] -
             car_corners[int(distance_to_img_center[k, 1]), 1]) ** 2)

        distance_to_img_center[k, 3] = car_corners[int(distance_to_img_center[k, 1]), 0]
        distance_to_img_center[k, 4] = car_corners[int(distance_to_img_center[k, 1]), 1]

    distance_to_img_center = np.array(sorted(distance_to_img_center, key=itemgetter(2)))

    # Estimated Centroid of bounding box
    measurement_vec[0] = (distance_to_img_center[1, 3] + distance_to_img_center[2, 3]) / 2
    measurement_vec[1] = (distance_to_img_center[1, 4] + distance_to_img_center[2, 4]) / 2
    # Estimated Orientation of the bounding box
    measurement_vec[2] = math.atan2(distance_to_img_center[0, 4] - distance_to_img_center[2, 4],
                                    distance_to_img_center[0, 3] - distance_to_img_center[2, 3])

    veh_length = distance_to_img_center[2, 2]
    veh_width = distance_to_img_center[1, 2]

    return veh_length, veh_width, measurement_vec, car_corners


def back_to_pixel(car_corners_kf, meter_to_px, orientation_offset,
                  linear_offsets):
    """
    Transforms the car corners in Meter back to pixels
    """
    car_corners_picture = np.zeros((4, 2))
    # Translating the corners of the bounding box
    # to compensate the linear offsets
    car_corners_picture[:, 0:2] = car_corners_kf[:, 0:2] + linear_offsets.T

    rotation_matrix = np.array([[math.cos(orientation_offset),
                                 - math.sin(orientation_offset)],
                                [math.sin(orientation_offset),
                                 math.cos(orientation_offset)]])

    # Rotating the corners of the bounding box to
    # compensate the orientation offset
    for idx_rotate_gcp in range(len(car_corners_picture)):
        car_corners_picture[idx_rotate_gcp, 0:2] = (
                rotation_matrix @ car_corners_picture[idx_rotate_gcp, 0:2].T).T

    # Rotating the corners of the bounding box 180 deg around x axis
    r_x = np.array([[1, 0], [0, -1]])
    for idx_rotate_gcp in range(len(car_corners_picture)):
        car_corners_picture[idx_rotate_gcp, 0:2] = (
                r_x @ car_corners_picture[idx_rotate_gcp, 0:2].T).T

    # applying the spatial resolution to the bounding box
    car_corners_picture[:, 0:2] = car_corners_picture[:, 0:2] / meter_to_px

    return car_corners_picture


def kalman_filter(fps, resolution, drone_height, measurement_vec, x_old, system_covariance_matrix,
                  stand_still, veh_l, veh_w, car_corners):
    """
    Kalman Filter

    Arguments:
        measurement_vec {Vector} -- Measurement Vector returned by length_class
        x_old {Vector} -- Previous state vector
        stand_still {float} -- threshold value for low speeds
        car_corners {Matrix} -- car corners returned by length_class

    Returns:
        new state vector, new system covariance matrix, extras that are
        not part of the state vector,car corners kf
    """
    if all([math.isfinite(x) for x in measurement_vec]):  # measurement found
        if math.sqrt(x_old[3] ** 2 + x_old[2] ** 2) > stand_still:
            # if the vehicle is moving, estimate the
            # course over ground (cog) from velocity over ground
            raw_cog = math.atan2(x_old[3], x_old[2])
        else:
            raw_cog = x_old[6]
        # Correcting the vehicle orientation --> Approaching the
        # Orientation to the cog
        # 90 deg to 270 deg
        if ((math.pi / 2) < max([measurement_vec[2], raw_cog]) - min([measurement_vec[2], raw_cog])) \
                and ((math.pi * 3 / 2) > max([measurement_vec[2], raw_cog]) - min(
            [measurement_vec[2], raw_cog])):
            if measurement_vec[2] - raw_cog > 0:
                measurement_vec[2] -= math.pi
            else:
                measurement_vec[2] += math.pi
        # 270 deg to 360 deg
        elif (math.pi * 3 / 2) < max([measurement_vec[2], raw_cog]) - min(
                [measurement_vec[2], raw_cog]):
            if (measurement_vec[2] - raw_cog) > 0:
                measurement_vec[2] -= 2 * math.pi
            else:
                measurement_vec[2] += 2 * math.pi
        # 0 deg to 90 deg
        else:
            measurement_vec[2] = measurement_vec[2]

        obs_model = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0]])
    else:
        # no measurement, infinite measurement vector
        measurement_vec = [999, 999, 999]
        obs_model = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0]])

    # Tuning parameters
    if resolution[0] == 3840:
        measurement_noise = np.diag([0.074, 0.075, 0.027])
        system_noise = np.diag([0.3, 0.3, .1, .1, 3, 3, 1, 1])

    elif resolution[0] == 1920 and drone_height <= 50:
        measurement_noise = np.diag([0.0149, 0.0149, 0.0583])
        system_noise = np.diag([.05, .05, .1, .1, 3, 3, 1, 1])

    elif resolution[0] == 1920 and 50 <= drone_height <= 75:
        measurement_noise = np.diag([0.0838, 0.0838, 0.0843])
        system_noise = np.diag([.104, .104, .1, .1, 3, 3, 1, 1])

    elif resolution[0] == 1920 and 75 <= drone_height <= 100:
        measurement_noise = np.diag([0.0838, 0.0838, 0.0843])
        system_noise = np.diag([.104, .104, .1, .1, 3, 3, 1, 1])
    else:
        if resolution[0] not in [1920, 3840]:
            measurement_noise = np.diag([0.0838, 0.0838, 0.0843])
            system_noise = np.diag([.104, .104, .1, .1, 3, 3, 1, 1])
            print("WARNING: KF tuning parameters not defined for this resolution."
                  " Setting to default 1920x1080 values.")
        else:
            print("WARNING: DroneHeight not supported."
                  " defaulting to maximum height of 100 meter")
    d_t = 1 / fps
    system_model = np.array([[1, 0, d_t, 0, (d_t ** 2) / 2, 0, 0, 0],
                             [0, 1, 0, d_t, 0, (d_t ** 2) / 2, 0, 0],
                             [0, 0, 1, 0, d_t, 0, 0, 0],
                             [0, 0, 0, 1, 0, d_t, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, d_t],
                             [0, 0, 0, 0, 0, 0, 0, 1]])

    x_pred = system_model @ x_old
    pn_n1 = (system_model @ system_covariance_matrix @ system_model.T + system_noise)
    k_gain = (pn_n1 @ obs_model.T) @ np.linalg.inv(
        obs_model @ pn_n1 @ obs_model.T + measurement_noise)
    x_new = x_pred + k_gain @ (measurement_vec - obs_model @ x_pred)

    system_covariance_matrix = ((np.eye(len(system_model)) - k_gain @ obs_model) @ pn_n1)

    extras_new, car_corners_kf = create_extras_and_kf_corners(x_old, x_new, cfg.standstill,
                                                              car_corners, veh_l, veh_w)

    return x_new, system_covariance_matrix, extras_new, car_corners_kf


def create_extras_and_kf_corners(x_old, x_new, stand_still, car_corners,
                                 veh_l, veh_w):
    """
    Estimates vehicle state variables that are not part of the state vector

        Calculates the corners of the bounding box out of :
            1. estimated state of kalman filter (position and orientation)
            2. width and length of the vehicle
    """

    extras_new = [0 for _ in range(6)]
    if math.sqrt(x_new[2] ** 2 + x_new[3] ** 2) < stand_still:
        extras_new[2] = x_old[6]
    else:
        extras_new[2] = math.atan2(x_new[3], x_new[2])

    extras_new[0] = math.sqrt(x_new[2] ** 2 + x_new[3] ** 2)  # velocity
    extras_new[1] = math.sqrt(x_new[4] ** 2 + x_new[5] ** 2)  # acceleration
    extras_new[3] = extras_new[2] - x_new[6]  # Sideslip
    acc = np.array(
        [[math.cos(-x_new[6]), - math.sin(-x_new[6])],
         [math.sin(-x_new[6]), math.cos(-x_new[6])]]) \
          @ np.array([[x_new[4]], [x_new[5]]])

    extras_new[4] = float(acc[0])
    extras_new[5] = float(acc[1])

    ex1 = math.cos(x_new[6])
    ex2 = math.sin(x_new[6])
    ey1 = - ex2
    ey2 = ex1
    car_corners_kf = np.zeros((np.matrix(car_corners).shape))
    # Corner 1
    car_corners_kf[0, 0] = x_new[0] + veh_l / 2 * ex1 - 0.5 * veh_w * ey1
    car_corners_kf[0, 1] = x_new[1] + veh_l / 2 * ex2 - 0.5 * veh_w * ey2

    # Corner 2
    car_corners_kf[1, 0] = x_new[0] + veh_l / 2 * ex1 + 0.5 * veh_w * ey1
    car_corners_kf[1, 1] = x_new[1] + veh_l / 2 * ex2 + 0.5 * veh_w * ey2

    # Corner 3
    car_corners_kf[2, 0] = x_new[0] - veh_l / 2 * ex1 + 0.5 * veh_w * ey1
    car_corners_kf[2, 1] = x_new[1] - veh_l / 2 * ex2 + 0.5 * veh_w * ey2

    # Corner 4
    car_corners_kf[3, 0] = x_new[0] - veh_l / 2 * ex1 - 0.5 * veh_w * ey1
    car_corners_kf[3, 1] = x_new[1] - veh_l / 2 * ex2 - 0.5 * veh_w * ey2

    car_corners_kf = np.c_[car_corners_kf, [1, 2, 3, 4]]
    return extras_new, car_corners_kf


class KF():
    """
    Kalman Filter Class that represents a KF Object for every Track
    Gets created on every new unassigned detection and deletes itself
    if track lost (invisible for too long)

    Also keeps track of old variables, since they may be needed for saving.
    """

    def __repr__(self):
        return str(self.car_corners)

    def __init__(self, k, boxes_rot, unassigned_detections):
        self._car_length = 0
        self._car_width = 0
        self.x_old = np.zeros((8,))
        self.system_covariance_matrix = np.eye(len(self.x_old))
        self.system_covariance_matrix[6, 6] = self.system_covariance_matrix[6, 6] * 0.0001
        self._X = np.zeros((8, 1))
        self._E = np.zeros((6, 1))
        self._car_corners = None
        self._car_corners_kf = None
        if len(boxes_rot[global_step]) >= 1:  # if the drone saw a vehicle
            kkk = unassigned_detections[k]
            self._car_corners = [
                [boxes_rot[global_step][kkk][0][0], boxes_rot[global_step][kkk][0][1]],
                [boxes_rot[global_step][kkk][1][0], boxes_rot[global_step][kkk][1][1]],
                [boxes_rot[global_step][kkk][2][0], boxes_rot[global_step][kkk][2][1]],
                [boxes_rot[global_step][kkk][3][0], boxes_rot[global_step][kkk][3][1]]]

        else:
            self._car_corners = np.zeros((4, 2))

        veh_l, veh_w, measurement_vec, car_corners = length_class(
            self.car_corners,
            [cfg.width, cfg.height],
            cfg.meter_to_px,
            cfg.orientation_offset,
            cfg.linear_offsets,
            cfg.drone_height,
            cfg.outer_height,
            cfg.inner_height,
            cfg.radial_displacement)

        x_new, system_covariance_matrix, extras_new, car_corners_kf = kalman_filter(
            cfg.fps,
            [cfg.width, cfg.height],
            cfg.drone_height,
            measurement_vec,
            self.x_old,
            self.system_covariance_matrix,
            cfg.standstill,
            veh_l,
            veh_w,
            car_corners, )

        self._car_corners_kf = car_corners_kf
        self._X = x_new
        self.x_old = x_new
        self._E = extras_new
        self.system_covariance_matrix = system_covariance_matrix
        self._car_length = veh_l
        self._car_width = veh_w
        self.history_X = []
        self.history_E = []
        self.history_car_corners = []
        self.history_car_corners_kf = []
        self.history_car_length = []
        self.history_car_width = []
        self.history_corresponding_frame = []

    def predict_new_location(self):
        invalid_Z = [float("nan") for _ in range(3)]
        x_new, system_covariance_matrix, \
        extras_new, car_corners_kf = kalman_filter(cfg.fps,
                                                   [cfg.width, cfg.height],
                                                   cfg.drone_height,
                                                   invalid_Z,
                                                   self.X,
                                                   self.system_covariance_matrix,
                                                   cfg.standstill,
                                                   self.car_length,
                                                   self.car_width,
                                                   self.car_corners)

        transformed_corners = back_to_pixel(car_corners_kf, cfg.meter_to_px,
                                            cfg.orientation_offset,
                                            cfg.linear_offsets)
        self.X = x_new
        self.x_new = x_new
        self.car_corners_kf = car_corners_kf
        self.car_corners = transformed_corners
        self.car_length = self.car_length
        self.car_width = self.car_width
        self.E = extras_new
        self.system_covariance_matrix = system_covariance_matrix

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self.history_X.append(self._X)
        self._X = value

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, value):
        self.history_E.append(self._E)
        self._E = value

    @property
    def car_corners(self):
        return self._car_corners

    @car_corners.setter
    def car_corners(self, value):
        self.history_car_corners.append(self._car_corners)
        self.history_corresponding_frame.append(global_step)
        self._car_corners = value

    @property
    def car_corners_kf(self):
        return self._car_corners_kf

    @car_corners_kf.setter
    def car_corners_kf(self, value):
        self.history_car_corners_kf.append(self._car_corners_kf)
        self._car_corners_kf = value

    @property
    def car_length(self):
        return self._car_length

    @car_length.setter
    def car_length(self, value):
        self.history_car_length.append(self._car_length)
        self._car_length = value

    @property
    def car_width(self):
        return self._car_width

    @car_width.setter
    def car_width(self, value):
        self.history_car_width.append(self._car_width)
        self._car_width = value


def bbox_to_polygon(box):
    """
    Converts bbox in the format [x1,y1,...x4,y4] into a shaply Polygon object
    """
    return Polygon(((box[0], box[1]), (box[2], box[3]), (box[4], box[5]), (box[6], box[7])))


def is_intersecting(box_a, box_b):
    """
    Checks if two bboxes intersect each other
    """
    p_1 = bbox_to_polygon(box_a)
    p_2 = bbox_to_polygon(box_b)
    return p_1.intersects(p_2)


def iou(box_a, box_b):
    """
    Return the iou of two bboxes
    """
    try:
        p_1 = bbox_to_polygon(box_a)
        p_2 = bbox_to_polygon(box_b)
        area_of_overlap = p_1.intersection(p_2).area

        area_of_union = p_1.area + p_2.area - area_of_overlap
        if area_of_union == 0:
            return 0
        iou = area_of_overlap / area_of_union
        return iou
    except:
        return 0


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.5):
    """
    Assigns detections to tracked object (both represented as rotated bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d_idx, det in enumerate(detections):
        for t_idx, trk in enumerate(trackers.get_tracks()):
            trk = trk[TrackEnum.KALMANFILTER].car_corners

            trk = np.array(trk).flatten()
            det = np.array(det).flatten()

            iou_matrix[d_idx, t_idx] = iou(det, trk)

    matched_indices = np.asarray(linear_sum_assignment(-iou_matrix)).T
    unmatched_detections = []
    for d_idx, det in enumerate(detections):
        if (d_idx not in matched_indices[:, 0]):
            unmatched_detections.append(d_idx)
    unmatched_trackers = []
    for t_idx, trk in enumerate(trackers.get_tracks()):
        if (t_idx not in matched_indices[:, 1]):
            unmatched_trackers.append(t_idx)
    # filter out matched with low IOU
    matches = []

    for m_idx in matched_indices:
        if (iou_matrix[m_idx[0], m_idx[1]] < iou_threshold):
            unmatched_detections.append(m_idx[0])
            unmatched_trackers.append(m_idx[1])
        else:
            matches.append(m_idx.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def detection_to_track_assignment(cost_of_non_assignment, n_veh):
    """
    Compute the cost of assigning each detection to each track
    """
    dets = []
    for i in range(n_veh):
        if len(_boxes_rot[global_step]) >= 1:  # -> drone saw a vehicle
            car_corners = np.array([_boxes_rot[global_step][i][x] for x in range(4)])
        else:  # -> drone saw no vehicle
            print("no vehicle")
            car_corners = np.zeros((4, 2))
        dets.append(car_corners)
    # check for multiple detections on same vehicle
    for k_idx, det1 in enumerate(dets):
        for l_idx, det2 in enumerate(dets):
            if k_idx == l_idx: continue
            if is_intersecting(det1.flatten(), det2.flatten()):
                dets[k_idx] = combine_detection(det1, det2)
                dets[l_idx] = np.zeros((4, 2))
                break

    return associate_detections_to_trackers(dets, tracks, cost_of_non_assignment)


def combine_detection(det1, det2):
    """
    If two detections intersect each other they will be combined since
    it`s most likely that they are the same object
    (if not the vehicles crashed)
    """
    p_1 = bbox_to_polygon(det1.flatten()).area
    p_2 = bbox_to_polygon(det2.flatten()).area

    return det1 if p_1 > p_2 else det2


def update_assigned_tracks(assignments):
    """
     assigned tracks -> runs kalman filter on the detections and appends new values to KF object
    """
    num_assigned_tracks = len(assignments)
    for k in range(num_assigned_tracks):

        track_idx = assignments[k, 1]
        det_idx = assignments[k, 0]

        if len(_boxes_rot[global_step]) >= 1:  # if the drone saw a vehicle
            tracks[track_idx][TrackEnum.KALMANFILTER].car_corners = [
                [_boxes_rot[global_step][det_idx][0][0],
                 _boxes_rot[global_step][det_idx][0][1]],
                [_boxes_rot[global_step][det_idx][1][0],
                 _boxes_rot[global_step][det_idx][1][1]],
                [_boxes_rot[global_step][det_idx][2][0],
                 _boxes_rot[global_step][det_idx][2][1]],
                [_boxes_rot[global_step][det_idx][3][0],
                 _boxes_rot[global_step][det_idx][3][1]]]

        else:
            track_idx[track_idx][TrackEnum.KALMANFILTER].car_corners = np.zeros((4, 2))

        veh_l, veh_w, measurement_vec, car_corners_length_class = length_class(
            tracks[track_idx][TrackEnum.KALMANFILTER].car_corners,
            [cfg.width, cfg.height],
            cfg.meter_to_px,
            cfg.orientation_offset,
            cfg.linear_offsets,
            cfg.drone_height,
            cfg.outer_height,
            cfg.inner_height,
            cfg.radial_displacement)

        x_new, system_covariance_matrix, extras_new, car_corners_kf = kalman_filter(
            cfg.fps,
            [cfg.width, cfg.height],
            cfg.drone_height,
            measurement_vec,
            tracks[track_idx][TrackEnum.KALMANFILTER].x_old,
            tracks[track_idx][TrackEnum.KALMANFILTER].system_covariance_matrix,
            cfg.standstill,
            veh_l,
            veh_w,
            car_corners_length_class)

        tracks[track_idx][TrackEnum.KALMANFILTER].X = x_new
        tracks[track_idx][TrackEnum.KALMANFILTER].x_old = x_new
        tracks[track_idx][TrackEnum.KALMANFILTER].E = extras_new
        tracks[track_idx][
            TrackEnum.KALMANFILTER].system_covariance_matrix = system_covariance_matrix
        tracks[track_idx][TrackEnum.KALMANFILTER].car_length = veh_l
        tracks[track_idx][TrackEnum.KALMANFILTER].car_width = veh_w
        tracks[track_idx][TrackEnum.KALMANFILTER].car_corners_kf = car_corners_kf
        tracks[track_idx][TrackEnum.AGE] += 1
        tracks[track_idx][TrackEnum.TOTALVISIBLECNT] += 1
        tracks[track_idx][TrackEnum.CONSECINVISCNT] = 0


def spot_interrupt():
    """
    Checks if all tracks are still valid
    (consecutive invisible count not too high)
    If not valid the corresponding track index gets passed to the
    delete_tracks function
    """
    if len(tracks) == 0:
        return
    lost_idx = []
    for i, trk in enumerate(tracks.get_tracks()):
        if trk[TrackEnum.CONSECINVISCNT] >= cfg.invisible_for_too_long:
            lost_idx.append(i)
    # delete lost tracks
    tracks.delete_tracks(lost_idx)


def update_unassigned_tracks(unassigned_tracks):
    """
    Updates the tracks with no detection mapping

    """
    for k in range(len(unassigned_tracks)):
        ind = unassigned_tracks[k]
        if not tracks[ind]: return
        tracks[ind][TrackEnum.AGE] += 1
        tracks[ind][TrackEnum.CONSECINVISCNT] += 1
        tracks[ind][TrackEnum.KALMANFILTER].predict_new_location()


def create_new_track(unassigned_detections):
    """
    Creates new KF instance for every unassigned detection

    """
    for k in range(len(unassigned_detections)):
        kf_obj = KF(k, _boxes_rot, unassigned_detections)
        tracks.create_entry_with_kf(kf_obj)


def process_finished_track(track):
    """
    processes the final track before deletion and adds it to a final dict which gets saved in the end
    """

    kf_obj = track[TrackEnum.KALMANFILTER]
    track_length = len(kf_obj.history_E)

    pos_x, pos_y, speed, vel_x_ltp, vel_y_ltp, acc_magn, acc_x_ltp, acc_y_ltp, acc_x_lcp, acc_y_lcp, \
    yaw, yaw_in_img, course_og, sideslip, veh_length, veh_width, car_cornerskf_tracker, \
    car_corners_tracker, corresponding_frame = ([0 for _ in range(track_length)] for _ in range(19))


    for idx in range(track_length):

        his_x = kf_obj.history_X[idx]
        his_e = kf_obj.history_E[idx]

        pos_x[idx] = (his_x[0])
        pos_y[idx] = (his_x[1])
        speed[idx] = (math.sqrt(his_x[2] ** 2 + his_x[3] ** 2))
        vel_x_ltp[idx] = (his_x[2]) # LTP: local tangent plane
        vel_y_ltp[idx] = (his_x[3])
        acc_magn[idx] = (math.sqrt(his_x[4] ** 2 + his_x[5] ** 2))
        acc_x_ltp[idx] = (his_x[4])
        acc_y_ltp[idx] = (his_x[5])
        acc_x_lcp[idx] = (his_x[4]) # LCP: local car plane
        acc_y_lcp[idx] = (his_x[5])
        yaw[idx] = (his_x[6])
        yaw_in_img[idx] = (his_x[6] - cfg.orientation_offset)
        course_og[idx] = (his_e[2])
        sideslip[idx] = (his_e[3])
        veh_length[idx] = (kf_obj.history_car_length[idx])
        veh_width[idx] = (kf_obj.history_car_width[idx])
        car_cornerskf_tracker[idx] = (kf_obj.history_car_corners_kf[idx].tolist())
        car_corners_tracker[idx] = (kf_obj.history_car_corners[idx])
        corresponding_frame[idx] = (kf_obj.history_corresponding_frame[idx])

    # Drop entries that have not fully entered the frame
    avg_veh_length = ((sum(veh_length) / len(veh_length)) / 100) * cfg.fully_entered_threshold

    idx_to_drop = []
    for idx in range(track_length):
        if not veh_length[idx] > avg_veh_length:
            idx_to_drop.append(idx)
    # Make Sure to drop only at the beginning and in the end
    first_range = []
    for i in range(len(idx_to_drop) -1):
        if idx_to_drop[i + 1] == idx_to_drop[i] +1:
            first_range.append(idx_to_drop[i])
        else:
            break

    second_range = []
    for i in range(len(idx_to_drop) -1)[::-1]:
        if idx_to_drop[i + -1] == idx_to_drop[i] + -1:
            second_range.append(idx_to_drop[i])
        else:
            break
    idx_to_drop = list(set(first_range + second_range))

    avg_speed = sum(speed) / len(speed)
    if avg_speed <= cfg.standstill:
        # Discard track if the average speed is below the standstill threshold
        return
    # Dropping indices  
    final_tracks[final_trk_id] = {
        "posX": [el for i, el in enumerate(pos_x) if i not in idx_to_drop],
        "posY": [el for i, el in enumerate(pos_y) if i not in idx_to_drop],
        "speed": [el for i, el in enumerate(speed) if i not in idx_to_drop],
        "vel_x": [el for i, el in enumerate(vel_x_ltp) if i not in idx_to_drop],
        "vel_y": [el for i, el in enumerate(vel_y_ltp) if i not in idx_to_drop],
        "acc_ltp" : [el for i, el in enumerate(acc_magn) if i not in idx_to_drop],
        "accX_LTP": [el for i, el in enumerate(acc_x_ltp) if i not in idx_to_drop],
        "accY_LTP": [el for i, el in enumerate(acc_y_ltp) if i not in idx_to_drop],
        "accX_LCP" : [el for i, el in enumerate(acc_x_lcp) if i not in idx_to_drop],
        "accY_LCP" : [el for i, el in enumerate(acc_y_lcp) if i not in idx_to_drop],
        "yaw": [el for i, el in enumerate(yaw) if i not in idx_to_drop],
        "yawInImg": [el for i, el in enumerate(yaw_in_img) if i not in idx_to_drop],
        "course_og": [el for i, el in enumerate(course_og) if i not in idx_to_drop],
        "sideslip" : [el for i, el in enumerate(sideslip) if i not in idx_to_drop],
        "veh_length": [el for i, el in enumerate(veh_length) if i not in idx_to_drop],
        "veh_width": [el for i, el in enumerate(veh_width) if i not in idx_to_drop],
        "carCornersKF_tracker": [el for i, el in enumerate(car_cornerskf_tracker) if i not in idx_to_drop],
        "carCorners_tracker": [el for i, el in enumerate(car_corners_tracker) if i not in idx_to_drop],
        "corresponding_frame": [el for i, el in enumerate(corresponding_frame) if i not in idx_to_drop]
    }

    globals()['final_trk_id'] += 1

def makedir_or_remove(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    else:
        if len(os.listdir(path)) != 0:
            for element in os.listdir(path):
                os.remove(path + element)

def save_tracks_csv(tracks, save_path):
    makedir_or_remove(save_path)

    for key, subdict in tracks.items():
        with open(str(Path(save_path) / "track_") + str(key) + ".csv",'w', newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(subdict.keys())
            writer.writerows(zip(*[subdict[key] for key in subdict.keys()]))


def save_tracks_json(tracks, save_path):
    import json

    makedir_or_remove(save_path)

    for key, value in tracks.items():
        with open(str(Path(save_path) / "track_") + str(key) + ".json", "w") as json_file:
            value = json.dumps(str(value))
            json.dump(value, json_file)


def save_tracks_pickle(tracks, save_path):
    import pickle

    makedir_or_remove(save_path)

    for key, value in tracks.items():
        with open(str(Path(save_path) / "track_") + str(key) + ".pkl", "wb") as pickle_file:
            pickle.dump(value, pickle_file)


def save_tracks_txt(tracks, save_path):
    makedir_or_remove(save_path)

    for key, value in tracks.items():
        with open(str(Path(save_path) / "track_") + str(key) + ".txt", "w") as txt_file:
            txt_file.write(str(value))


def final_clean_up(save_dir, visualize):
    """
    Final Cleanup after the last iteration
    Saves all the remaining tracks with a valid min_hits amount
    Writes the final tracks to disk
    """

    [process_finished_track(trk) for trk in tracks.get_tracks()
     if trk[TrackEnum.CONSECINVISCNT] <= cfg.invisible_for_too_long
     and trk[TrackEnum.TOTALVISIBLECNT] >= cfg.min_hits]

    # Sort tracks based on frame
    all_tracks = [track for track in globals()["final_tracks"].items()]
    all_tracks = sorted(all_tracks, key=lambda x : x[1]["corresponding_frame"][0])
    final_tracks = {}
    for i,sorted_track in enumerate(all_tracks):
        final_tracks[i] = sorted_track[1]

    save_dir = os.path.abspath(save_dir)

    if cfg.save_as_csv:
        csv_dir = str(Path(save_dir).parents[0]) + "/results/csv/"
        save_tracks_csv(final_tracks, csv_dir)

    if cfg.save_as_json:
        json_dir = str(Path(save_dir).parents[0]) + "/results/json/"
        save_tracks_json(final_tracks, json_dir)

    if cfg.save_as_pickle:
        pickle_dir = str(Path(save_dir).parents[0]) + "/results/pickle/"
        save_tracks_pickle(final_tracks, pickle_dir)

    if cfg.save_as_txt:
        txt_dir = str(Path(save_dir).parents[0]) + "/results/txt/"
        save_tracks_txt(final_tracks, txt_dir)

    if visualize:
        visualize_tracks(Path(save_dir), final_tracks, cfg)

    print("finished video -> saved a total of {} track(s)\n".format(globals()['final_trk_id']),
          file=sys.__stdout__)

    # resetting global track variables -> necessary if there's more than one video
    globals()["final_tracks"].clear()
    globals()["final_trk_id"] = 0

    # Deleting the preprocessed_images folder, if 'del_temp_images' set to True
    if cfg.del_temp_images:
        im_path = (save_dir + "/preprocessed_images/")
        [os.remove(im_path + file) for file in os.listdir(im_path)]


def postprocess(temp_dir, config):
    global cfg
    cfg = config

    global final_tracks
    global final_trk_id
    global _boxes_rot
    global global_step
    global tracks

    tracks = Tracks()
    final_tracks = {}
    final_trk_id = 0

    _boxes_rot = np.load(temp_dir + "data/boxes.npy", allow_pickle=True)

    if cfg.meter_to_px == 1:
        print("WARNING: the 'meter_to_pixel' value is set to 1,"
              " so the results are based on pixel and not meters")

    results_dir = "./" + str(Path(temp_dir).parents[0]) + "/results/"

    if os.path.exists(results_dir) and len(os.listdir(results_dir)) != 0 and not cfg.visualize:
        print("found existing results -> skipping postprocessing")
        return
    if cfg.visualize and len(
            [el for el in os.listdir(str(Path(results_dir).parents[0]))
             if el.endswith(".mp4")]) != 0:
        print("found existing track visualization video -> skipping"
              " postprocessing and visualization video")
        return

    for global_step in trange(len(_boxes_rot), disable=not cfg.verbose,
                              desc="Postprocessing".center(30)):
        if _boxes_rot[global_step] == 0:
            print("no detection")
            continue

        n_veh = len(_boxes_rot[global_step])
        assignments, unassigned_detections, unassigned_tracks = detection_to_track_assignment(
            cfg.cost_of_non_assignment,
            n_veh, )

        create_new_track(unassigned_detections)
        update_assigned_tracks(assignments)
        update_unassigned_tracks(unassigned_tracks)
        spot_interrupt()

    final_clean_up(temp_dir, cfg.visualize)
