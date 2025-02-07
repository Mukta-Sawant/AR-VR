import cv2
import json
import os

import numpy as np
from math import pi
from rendering import convert_model_to_camera_space
from rendering import project_to_image
from rendering import render_image
from rendering import render_wireframe
from rendering_helpers import make_intrinsics, yp_mat, compare_images

### CAMERA SPACE

def write_test_camera_space(reference_file, test_name, model_file, translation, yaw, pitch):
    rotation = yp_mat(yaw, pitch)
    with open(model_file, 'r') as f:
        model = json.load(f)

    vertices = np.array(model["vertices"])
    points = convert_model_to_camera_space(vertices, rotation, translation)
    np.save(reference_file, points)

def test_camera_space(reference_file, test_name, model_file, translation, yaw, pitch):
    rotation = yp_mat(yaw, pitch)
    with open(model_file, 'r') as f:
        model = json.load(f)

    vertices = np.array(model["vertices"])
    actual_points = convert_model_to_camera_space(vertices, rotation, translation)

    expected_points = np.load(reference_file)
    comps = np.isclose(expected_points, actual_points)
    if comps.all():
        print("Test '" + test_name + "' passed")
    else:
        print("Test '" + test_name + "' failed, " + str(comps.sum()) + " of " + str(comps.size))


test_camera_space(
    reference_file="tests/cs_simple_square.npy",
    test_name="Camera Space - Simple Square Identity",
    model_file="models/square.json",
    translation=(0, 0, 0),
    yaw=0,
    pitch=0,
)

test_camera_space(
    reference_file="tests/cs_square_negz.npy",
    test_name="Camera Space - Simple Square Negative Z",
    model_file="models/square.json",
    translation=(0, 0, -5),
    yaw=0,
    pitch=0,
)

test_camera_space(
    reference_file="tests/cs_square_translation.npy",
    test_name="Camera Space - Square Translation",
    model_file="models/square.json",
    translation=(1.5, 1, -5),
    yaw=0,
    pitch=0,
)

test_camera_space(
    reference_file="tests/cs_rotation.npy",
    test_name="Camera Space - Square Rotation",
    model_file="models/square.json",
    translation=(1.5, 1, -5),
    yaw=-0.15,
    pitch=0.11,
)

# PROJECT TO IMAGE

def write_test_projected_image(reference_file, test_name, model_file, translation, yaw, pitch, image_width, image_height, focal_length):
    rotation = yp_mat(yaw, pitch)
    camera_intrinsics = make_intrinsics(focal_length, image_width, image_height)

    with open(model_file, 'r') as f:
        model = json.load(f)

    vertices = np.array(model["vertices"])
    camera_space_vertices = convert_model_to_camera_space(vertices, rotation, translation)
    projected_points = project_to_image(camera_space_vertices, camera_intrinsics)

    np.save(reference_file, projected_points)

def test_projected_image(reference_file, test_name, model_file, translation, yaw, pitch, image_width, image_height, focal_length):
    rotation = yp_mat(yaw, pitch)
    camera_intrinsics = make_intrinsics(focal_length, image_width, image_height)

    with open(model_file, 'r') as f:
        model = json.load(f)

    vertices = np.array(model["vertices"])
    camera_space_vertices = convert_model_to_camera_space(vertices, rotation, translation)
    actual_projected_points = project_to_image(camera_space_vertices, camera_intrinsics)

    expected_projected_points = np.load(reference_file, allow_pickle=True)
    match_count = 0
    total_count = len(expected_projected_points)

    for expected, actual in zip(expected_projected_points, actual_projected_points):
        if expected is None and actual is None:
            match_count += 1
        elif expected is not None and actual is not None and np.isclose(expected, actual).all():
            match_count += 1

    if match_count == total_count:
        print(f"Test '{test_name}' passed")
    else:
        print(f"Test '{test_name}' failed, {match_count} of {total_count} points matched")


test_projected_image(
    reference_file="tests/proj_simple_square.npy",
    test_name="Projection - Simple Square Identity",
    model_file="models/square.json",
    translation=np.array([0, 0, 0]), 
    yaw=0,
    pitch=0,
    image_width=512,
    image_height=512,
    focal_length=500
)

test_projected_image(
    reference_file="tests/proj_rotation.npy",
    test_name="Projection - Square Rotation",
    model_file="models/square.json",
    translation=np.array([1.5, 1, -5]),
    yaw=-0.15,
    pitch=0.11,
    image_width=512,
    image_height=512,
    focal_length=500
)

# RENDER IMAGE

def write_test_render_image(reference_file, test_name, model_file, translation, yaw, pitch, image_width, image_height, focal_length):
    os.makedirs(os.path.dirname(reference_file), exist_ok=True)

    rotation = yp_mat(yaw, pitch)
    camera_intrinsics = make_intrinsics(focal_length, image_width, image_height)

    with open(model_file, 'r') as f:
        model = json.load(f)

    image = render_wireframe(
        model=model,
        rotation=rotation,
        translation=translation,
        camera_intrinsics=camera_intrinsics,
        image_width=image_width,
        image_height=image_height
    )

    cv2.imwrite(reference_file, image)
    print(f"Reference wireframe image saved to {reference_file}")

def test_render_image(reference_file, test_name, model_file, translation, yaw, pitch, image_width, image_height, focal_length):
    if not os.path.exists(reference_file):
        print(f"Reference file '{reference_file}' not found. Generating it now...")
        write_test_render_image(reference_file, test_name, model_file, translation, yaw, pitch, image_width, image_height, focal_length)

    rotation = yp_mat(yaw, pitch)
    camera_intrinsics = make_intrinsics(focal_length, image_width, image_height)

    with open(model_file, 'r') as f:
        model = json.load(f)

    actual_image = render_wireframe(
        model=model,
        rotation=rotation,
        translation=translation,
        camera_intrinsics=camera_intrinsics,
        image_width=image_width,
        image_height=image_height
    )

    expected_image = cv2.imread(reference_file)

    if compare_images(expected_image, actual_image):
        print(f"Test '{test_name}' passed")
    else:
        print(f"Test '{test_name}' failed")

test_render_image(
    reference_file="tests/render_simple_square.png",
    test_name="Render - Simple Square",
    model_file="models/square.json",
    translation=np.array([0, 0, -5]),
    yaw=0,
    pitch=0,
    image_width=512,
    image_height=512,
    focal_length=500
)

test_render_image(
    reference_file="tests/render_rotation_square.png",
    test_name="Render - Rotated Square",
    model_file="models/square.json",
    translation=np.array([1.5, 1, -5]),
    yaw=-0.15,
    pitch=0.11,
    image_width=512,
    image_height=512,
    focal_length=500
)




## INTEGRATION

def write_test_image(reference_file, test_name, model_file, translation, yaw, pitch, image_width, image_height, focal_length):
    rotation = yp_mat(yaw, pitch)
    camera_intrinsics = make_intrinsics(focal_length, image_width, image_height)
    with open(model_file, 'r') as f:
        model = json.load(f)

    image = render_wireframe(
        model=model,
        rotation=rotation,
        translation=translation,
        camera_intrinsics=camera_intrinsics,
        image_width=image_width,
        image_height=image_height
    )
    cv2.imwrite(reference_file, image)


def test_image(reference_file, test_name, model_file, translation, yaw, pitch, image_width, image_height, focal_length):
    rotation = yp_mat(yaw, pitch)
    camera_intrinsics = make_intrinsics(focal_length, image_width, image_height)
    with open(model_file, 'r') as f:
        model = json.load(f)

    actual_image = render_wireframe(
        model=model,
        rotation=rotation,
        translation=translation,
        camera_intrinsics=camera_intrinsics,
        image_width=image_width,
        image_height=image_height
    )
    expected_image = cv2.imread(reference_file)
    if compare_images(expected_image, actual_image):
        print("Test '" + test_name + "' passed")
    else:
        print("Test '" + test_name + "' failed")

test_image(
    reference_file="tests/simple_square.png",
    test_name="Simple Square",
    model_file="models/square.json",
    yaw = 0,
    pitch = 0,
    translation = np.array([0, 0, -5.]),
    image_width = 512,
    image_height = 512,
    focal_length = 500
)

test_image(
    reference_file="tests/simple_cube.png",
    test_name="Simple Cube",
    model_file="models/cube.json",
    yaw = 0,
    pitch = 0,
    translation = np.array([0, 0, -5.]),
    image_width = 512,
    image_height = 512,
    focal_length = 500
)

test_image(
    reference_file="tests/simple_xyz.png",
    test_name="XYZ model test",
    model_file="models/xyz.json",
    yaw = 0,
    pitch = 0,
    translation = np.array([0, 0, -5.]),
    image_width = 512,
    image_height = 512,
    focal_length = 500
)

test_image(
    reference_file="tests/translate_square.png",
    test_name="Square with Translation",
    model_file="models/square.json",
    yaw = 0,
    pitch = 0,
    translation = np.array([0.5, 0.3, -5.]),
    image_width = 512,
    image_height = 512,
    focal_length = 500
)

test_image(
    reference_file="tests/rotate_square.png",
    test_name="Square with Rotation",
    model_file="models/square.json",
    yaw = 0.1,
    pitch = 0.1,
    translation = np.array([0, 0, -5.]),
    image_width = 512,
    image_height = 512,
    focal_length = 500
)

test_image(
    reference_file="tests/full_motion_square.png",
    test_name="Square with Rotation",
    model_file="models/square.json",
    yaw = 0.1,
    pitch = 0.1,
    translation = np.array([0.3, -0.5, -4]),
    image_width = 512,
    image_height = 512,
    focal_length = 500
)

test_image(
    reference_file="tests/image_size.png",
    test_name="Smaller image size",
    model_file="models/cube.json",
    yaw = 0.0,
    pitch = 0.0,
    translation = np.array([0.0, 0.0, -5]),
    image_width = 360,
    image_height = 640,
    focal_length = 600
)

test_image(
    reference_file="tests/cube_clipping.png",
    test_name="Cube clipping View",
    model_file="models/cube.json",
    yaw=pi / 6,
    pitch=pi / 12,
    translation=np.array([0, 0, -3.]),
    image_width=512,
    image_height=512,
    focal_length=500
)