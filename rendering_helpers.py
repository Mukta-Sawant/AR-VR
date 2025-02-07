from math import cos, sin, pi

import cv2
import numpy as np

def yp_mat(yaw, pitch):
    yaw_mat = np.array([[cos(yaw), 0, sin(yaw)],
                        [0, 1, 0],
                        [-sin(yaw), 0, cos(yaw)]])
    pitch_mat = np.array([[1, 0, 0],
                          [0, cos(pitch), -sin(pitch)],
                          [0, sin(pitch), cos(pitch)]])
    return yaw_mat @ pitch_mat

def clamp_pitch(pitch):
    return max(-pi/2, min(pitch, pi/2))


def make_intrinsics(focal_length, image_width, image_height):
    return np.array([[focal_length, 0., image_width/2], [0, -focal_length, image_height/2], [0, 0, 1]])

def compare_images(expected_image, actual_image):
    if (expected_image == actual_image).all():
        return True
    eb, eg, er = cv2.split(expected_image)
    ab, ag, ar = cv2.split(actual_image)
    cv2.imshow('Image Comparison (green = expected, blue = actual)', cv2.merge((ab, eg, ar & er)))
    cv2.waitKey(0)
    return False


# imgdiff(np1, np2)