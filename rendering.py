import json
import numpy as np
import cv2
from math import pi
from rendering_helpers import yp_mat, clamp_pitch, make_intrinsics


def convert_model_to_camera_space(vertices, camera_rotation, camera_translation):
    """
    Rotates and moves the model to move it into camera space.
    :param vertices: a list of 3D points formatted as 3-element lists
    :param camera_rotation: 3x3 numpy array representing the rotation matrix of the camera within world space.
    :param camera_translation: 3-element numpy array representing the position of the camera within world space.
    :return: a list the same size as vertices where the points are given relative to the camera's coordinate system
    """
    # Transform vertices from world to camera space
    vertices = np.array(vertices)
    vertices = vertices - camera_translation  # First translate
    vertices = np.dot(camera_rotation.T, vertices.T).T  # Then rotate with transpose
    
    return vertices

def project_to_image(vertices_camera_space, camera_intrinsics):
    """
    Project down the vertices within camera space into 2D pixel locations on the infinite image plane.
    Be sure to convert any points behind the camera to None.
    :param vertices_camera_space: List of 3D points indicating the vertices locations within camera space.
    :param camera_intrinsics: Camera matrix defined by focal length and centroid
    :return: A list of 2D points representing the pixel coordinates of projected vertices. If a vertex is behind the camera, it is replaced with None.
    """

    # Project vertices onto the image plane
    projected_points = []
    for vertex in vertices_camera_space:
        if vertex[2] <= 0:  # Check if point is behind camera
            projected_points.append(None)
            continue
    
        projected = camera_intrinsics @ vertex
        x, y = projected[:2] / projected[2]  # Normalize by depth
        projected_points.append((int(x), int(y)))
    

# code for debugging
    # debugged code 2
    # print("\nCamera Intrinsics Matrix:")
    # print(camera_intrinsics)
    
    # projected_points = []
    # for i, vertex in enumerate(vertices_camera_space):
    #     print(f"\nVertex {i}:")
    #     print(f"Camera space coordinates: {vertex}")
        
    #     if vertex[2] <= 0:
    #         print(f"Point {i} is behind camera")
    #         projected_points.append(None)
    #         continue

    #     # Try the calculation step by step
    #     x, y, z = vertex
        
    #     # Step 1: Perspective division
    #     x_normalized = x/z
    #     y_normalized = y/z
    #     homo_coords = np.array([x_normalized, y_normalized, 1.0])
    #     print(f"After perspective division: {homo_coords}")
        
    #     # Step 2: Apply camera intrinsics
    #     projected = np.dot(camera_intrinsics, homo_coords)
    #     print(f"After camera intrinsics: {projected}")
        
    #     projected_points.append(projected[:2])

    #     expected_points = np.load("tests/proj_rotation.npy", allow_pickle=True)
    #     print("Expected points from file:")
    #     print(expected_points)

    return projected_points

def render_image(image_space_vertices, edges, image_width, image_height):
    """
    Renders a wireframe model onto a blank image using given 2D projected vertices and edges.

    :param image_space_vertices: List of 2D points representing projected vertices in image space.
      Some values may be None if the corresponding 3D points were behind the camera.
    :param edges: List of pairs of indices indicating the connections between vertices.
    :param image_width: Width of the output image in pixels.
    :param image_height: Height of the output image in pixels.
    :return: A NumPy array representing the rendered wireframe image with a black background and white lines.
    """
    # Initialize a blank image
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Draw the edges
    for edge in edges:
        point1 = image_space_vertices[edge[0]]
        point2 = image_space_vertices[edge[1]]
        if point1 is not None and point2 is not None:
            point1 = (int(point1[0]), int(point1[1]))
            point2 = (int(point2[0]), int(point2[1]))
            cv2.line(image, point1, point2, (255, 255, 255), 1)

    return image

def render_wireframe(
    model,
    rotation, translation,
    camera_intrinsics,
    image_width, image_height,
):
    """
    Render a white wireframe model on a black background using given camera parameters.

    :param model: Dictionary representing the model to render. There are two entries:
      - "vertices" as a list of 3D points formatted as 3-element lists
      - "edges" as a list of pairs of indices into the "vertices" list
    :param rotation: 3x3 numpy array representing the rotation matrix of the camera within world space.
      Multiplying a point by this matrix converts from camera space to world space.
    :param translation: 3-element numpy array representing the position of the camera within world space.
    :param camera_intrinsics: 3x3 numpy array representing the camera intrinsics using the pinhole camera model.
    :param image_width: width of the image in pixels
    :param image_height: height of the image in pixels
    :return: A wireframe image of the wireframe model as viewed from the specified camera in the format of a numpy array.
    """

    vertices = np.array(model["vertices"])
    edges = model["edges"]

    # Step 1: Convert from world space to camera space
    vertices_camera_space = convert_model_to_camera_space(vertices, rotation, translation)

    # Step 2: Compress to the image plane
    points_2d = project_to_image(vertices_camera_space, camera_intrinsics)

    # Step 3: Color the pixels
    return render_image(points_2d, edges, image_width, image_height)


## Show a render

if __name__ == "__main__":
    # Initial values
    yaw = 0 # note, these are in radians
    pitch = 0 # note, these are in radians
    translation = np.array([0, 0, -5.])
    image_width = image_height = 512
    focal_length = 500
    # model_file = "models/xyz.json"
    # model_file = "models/cube.json"
    model_file = "models/square.json"

    # Settings
    TRANSLATION_STEP_SIZE = 0.2
    ROTATION_STEP_SIZE = pi / 18
    FOCAL_FACTOR = 1.1

    # Initalize dependent values
    rotation = yp_mat(yaw, pitch)
    camera_intrinsics = make_intrinsics(focal_length, image_width, image_height)
    with open(model_file, 'r') as f:
        model = json.load(f)

    # Render the image
    image = render_wireframe(
        model=model,
        rotation=rotation,
        translation=translation,
        camera_intrinsics=camera_intrinsics,
        image_width=image_width,
        image_height=image_height
    )
    cv2.imshow("Wireframe", image)
    
    # Show the image until a key is pressed
    keypress = cv2.waitKey()
