import cv2
import json
import os
import numpy as np
from math import pi
from rendering import convert_model_to_camera_space, project_to_image, render_image, render_wireframe
from rendering_helpers import make_intrinsics, yp_mat, compare_images

def write_test_camera_space(reference_file, test_name, model_file, translation, yaw, pitch):
    rotation = yp_mat(yaw, pitch)
    with open(model_file, 'r') as f:
        model = json.load(f)

    vertices = np.array(model["vertices"])
    points = convert_model_to_camera_space(vertices, rotation, translation)
    np.save(reference_file, points)

def test_camera_space(reference_file, test_name, model_file, translation, yaw, pitch):
    # First check if reference file exists, if not create it
    if not os.path.exists(reference_file):
        print(f"Reference file '{reference_file}' not found. Generating it now...")
        os.makedirs(os.path.dirname(reference_file), exist_ok=True)
        write_test_camera_space(reference_file, test_name, model_file, translation, yaw, pitch)

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
    # First check if reference file exists, if not create it
    if not os.path.exists(reference_file):
        print(f"Reference file '{reference_file}' not found. Generating it now...")
        os.makedirs(os.path.dirname(reference_file), exist_ok=True)
        write_test_projected_image(reference_file, test_name, model_file, translation, yaw, pitch, image_width, image_height, focal_length)

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

def run_test_case_1():
    # Test Case 1: Simple cube scale test (just farther away)
    test_camera_space(
        reference_file="tests/cs_scaled_cube.npy",
        test_name="Camera Space - Scaled Cube",
        model_file="models/cube.json",
        translation=(0, 0, -8),  # Farther away to appear smaller
        yaw=0,                   # No rotation
        pitch=0
    )

    test_projected_image(
        reference_file="tests/proj_scaled_cube.npy",
        test_name="Projection - Scaled Cube",
        model_file="models/cube.json",
        translation=(0, 0, -8),
        yaw=0,
        pitch=0,
        image_width=512,
        image_height=512,
        focal_length=500
    )

    test_render_image(
        reference_file="tests/render_scaled_cube.png",
        test_name="Render - Scaled Cube",
        model_file="models/cube.json",
        translation=np.array([0, 0, -8]),
        yaw=0,
        pitch=0,
        image_width=512,
        image_height=512,
        focal_length=500
    )

def run_test_case_2():
    # Test Case 2: Simple shift test (square shifted up and right)
    test_camera_space(
        reference_file="tests/cs_shifted_square.npy",
        test_name="Camera Space - Shifted Square",
        model_file="models/square.json",
        translation=(1, 1, -4),  # Shifted right and up
        yaw=0,                   # No rotation
        pitch=0
    )

    test_projected_image(
        reference_file="tests/proj_shifted_square.npy",
        test_name="Projection - Shifted Square",
        model_file="models/square.json",
        translation=(1, 1, -4),
        yaw=0,
        pitch=0,
        image_width=512,
        image_height=512,
        focal_length=500
    )

    test_render_image(
        reference_file="tests/render_shifted_square.png",
        test_name="Render - Shifted Square",
        model_file="models/square.json",
        translation=np.array([1, 1, -4]),
        yaw=0,
        pitch=0,
        image_width=512,
        image_height=512,
        focal_length=500
    )

if __name__ == "__main__":
    # Make sure the tests directory exists
    os.makedirs("tests", exist_ok=True)
    
    print("\nRunning Test Case 1: Simple Cube Scale Tests...")
    run_test_case_1()
    
    print("\nRunning Test Case 2: Simple Square Shift Tests...")
    run_test_case_2()