'''
Project By: Noga Levy ID: 315260927
            Tali Kalev ID: 208629691
Course: Advanced Python
Date: March 31, 2024

OS: Windows 11
Python Version: 3.11.7
Modules Used: NumPy, PIL
'''

import numpy as np
from PIL import Image, ImageDraw
from collections import namedtuple

# ====================MACROS====================
SCALE        = 250
IMAGE_HEIGHT = 1080
IMAGE_WIDTH  = 1920
IMAGE_CENTER = [IMAGE_WIDTH/2, IMAGE_HEIGHT/2]
XYZ_VALUES   = namedtuple("XYZ_VALUES", ["x", "y", "z"], defaults=(0,))

# ====================GIVEN DATA====================
CAMERA_POS        = XYZ_VALUES(0.736, -0.585, 0.338)
LIGHT_SOURCE      = XYZ_VALUES(0.563, 0.139, 0.815)
BACKGROUND_COLOUR = np.array([0.5, 0.5, 0.5]) * 255 # PIL need 0-255 values and not 0-1 rgb values # RGB

class CubeData():
    '''
    This Class contains all the data pertaining to a cube in the image including: position, rotation and colour
    '''
    def __init__(self, name, position, rotation, colour):
        self.name     = name
        self.position = position
        self.rotation = rotation
        self.colour   = colour

CUBE1 = CubeData(name="CUBE1", position=XYZ_VALUES(1.58, 0.08, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-57.7)), colour=np.array([0.6, 0.6, 0.6]) * 255) # GREY
CUBE2 = CubeData(name="CUBE2", position=XYZ_VALUES(0.58, -1.2, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-55.9)), colour=np.array([0.75, 0.5, 0.2]) * 255) # ORANGE
CUBE3 = CubeData(name="CUBE3", position=XYZ_VALUES(-0.75, 0.43, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-56.5)), colour=np.array([0.7, 0.3, 0.7]) * 255) # PURPLE
CUBE4 = CubeData(name="CUBE4", position=XYZ_VALUES(0.67, 1.15, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(0)), colour=np.array([0.47, 0.8, 0.42]) * 255) # GREEN

# ====================MATRICES CONSTANTS====================
# Define the camera's view direction (Z' vector from CAMERA_XYZ matrix)
CAMERA_VIEW_DIRECTION = np.array([CAMERA_POS.x, CAMERA_POS.y, CAMERA_POS.z])

# This is used to convert a 3D point to a 2D point
PROJECTION_MATRIX = np.array([
    [1, 0, 0],
    [0, 1, 0],
])

# Calculated manually using the instructions given. Y' is a vector that is perpendicular with the -Z' vector.
# The X' vector is perpendicular to the Z axis and was calculated by the cross product of the Y' and Z' (X' = Y' x Z').
CAMERA_XYZ = np.array([
    [0.621, 0.782, 0], # X'
    [0.265, -0.211, -0.941], # Y' 
    [0.736, -0.585, 0.338], # Z'
])

# Indexed corners of a cube identifying a face
FACES = [
    [0, 1, 2, 3], # Bottom
    [5, 6, 7, 4], # Top
    [0, 1, 5, 4], # Front
    [2, 3, 7, 6], # Back
    [0, 3, 7, 4], # Left
    [1, 2, 6, 5], # Right
]

POINTS = np.array([[-0.5, -0.5, -0.5],  # index 0
                   [0.5, -0.5, -0.5],   # index 1
                   [0.5, 0.5, -0.5],    # index 2
                   [-0.5, 0.5, -0.5],   # index 3
                   [-0.5, -0.5, 0.5],   # index 4
                   [0.5, -0.5, 0.5],    # index 5
                   [0.5, 0.5, 0.5],     # index 6
                   [-0.5, 0.5, 0.5]])   # index 7

# ====================CALCULATION MATRICES====================

def get_rotation_mat_z(angle):
    '''
    Function to get the roation matrix by given angle for z axis
    '''
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])

# ====================IMAGE CREATION FUNCTIONS==================== 
def create_image(height, width):
    '''
    Receives height and width of image
    Returns image with background colour represented as a 2D matrix
    '''
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[0:height, 0:width] = BACKGROUND_COLOUR
    return image

def create_cube(cube_center, rotation_angle, cube_colour, image_pil): 
    '''
    Main function to create cubes in 2D image given 3D position and rotation.
    Receives center position of cube, rotation angle, cube colour and image on which we are projecting.
    '''
    # Precompute the rotation matrix and projection matrix
    cube_rotation = get_rotation_mat_z(rotation_angle.z)
    projection_matrix_camera = np.dot(PROJECTION_MATRIX, CAMERA_XYZ)

    # Vectorized rotation and translation of points
    rotated_points = np.dot(POINTS, cube_rotation.T)
    translated_points = rotated_points + cube_center

    # Vectorized projection from 3D to 2D
    projected2d = np.dot(translated_points, projection_matrix_camera.T)

    # Vectorized scaling and centering of points
    projected_points = np.rint(projected2d[:, :2] * SCALE + IMAGE_CENTER).astype(int)

    # Draw the faces of the cube
    draw_faces_of_cube(translated_points, cube_center, projected_points, cube_colour, image_pil)


def find_face_normal_vector(points, face_indices, cube_center):
    '''
    Receives transformed 3D points of cube, 4 indices representing points on a face and the center of the cube.
    Returns the normal vector of the face.
    '''
    # Retrieve relevant points according to indices
    p1, p2, p3, p4 = points[face_indices]
    # Find midpoint of the cube face
    midpoint = (p2 + p4) / 2
    # Calculate vector from middle of cube to cube face
    vector_p = midpoint - cube_center
    # Normalize the vector using np.linalg.norm
    return vector_p /  np.linalg.norm(vector_p)  # return normalized vector
    
def draw_face(image_pil, face_points_2d, cube_colour, face_normal_vector):
    '''
    Receives: image to draw on, 4 2D points, cube colour, and the normal vector of the cube's face.
    Draws the face of the cube represented by the 4 points onto the image taking into account the light source
    to change the shading on the cube face accordingly.
    '''
    # Calculate the dot product of the face normal vector and the light source
    light_intensity = np.dot(face_normal_vector, LIGHT_SOURCE)
    # Calculate the intensity of the light on the face
    I = 0.15 + 0.85 * np.clip(light_intensity, 0, 1)
    # Apply the light intensity to the cube colour
    shaded_colour = (cube_colour * I).astype(int)
    # Convert the shaded colour to a tuple
    shaded_colour_tuple = tuple(shaded_colour.tolist())
    
    # Convert face_points_2d from a NumPy array to a list of tuples
    face_points_list = [tuple(point) for point in face_points_2d]

    # Draw the face of the cube using the PIL library
    draw = ImageDraw.Draw(image_pil)
    draw.polygon(face_points_list, fill=shaded_colour_tuple)

def draw_faces_of_cube(transformed_points_3d, cube_center, projected_points, cube_colour, image_pil):
    '''
    Using the 3D transformed points, for each face of the cube we decide if it faces
    the angle of the camera and if it does, draws it in our image using the 2D points.
    '''
    for face_indices in FACES:
        face_normal_vector = find_face_normal_vector(transformed_points_3d,face_indices, cube_center)
        # Check if face is in the direction of camera view
        if (np.dot(face_normal_vector, CAMERA_VIEW_DIRECTION) > 0):
            face_points_2d = [projected_points[i] for i in face_indices]
            draw_face(image_pil, face_points_2d, cube_colour, face_normal_vector)

def distance_from_camera(camera_pos, cube_pos):
    '''
    Function to calculate the distance from the camera to a cube using the cube center point.
    '''
    return np.sqrt((camera_pos.x - cube_pos.x) ** 2 + 
                   (camera_pos.y - cube_pos.y) ** 2 + 
                   (camera_pos.z - cube_pos.z) ** 2)

def draw_order(camera_pos, cubes):
    '''
    Function to determine the order in which the cubes are drawn.
    Returns list of CubeData objects where the first item should be drawn first as it is the furthest away
    from the camera and the last object is the closest.
    '''
    # Calculate the distance from the camera to each cube and store it with the cube's identifier
    distances = [(cube, distance_from_camera(camera_pos, cube.position)) for cube in cubes]
    # Sort the cubes by their distance from the camera (furthest to closest)
    sorted_cubes = sorted(distances, key=lambda x: x[1], reverse=True)
    # Return the order in which the cubes should be drawn
    return [cube[0] for cube in sorted_cubes]

# ====================MAIN FUNCTION====================
if __name__ == "__main__":
    # Creating size + background
    image_np = create_image(IMAGE_HEIGHT, IMAGE_WIDTH)
    image_pil = Image.fromarray(image_np, 'RGB')

    # Find the order in which to draw the cubes depending on distance from camera
    cubes = [CUBE1, CUBE2, CUBE3, CUBE4]
    order_to_draw = draw_order(CAMERA_POS, cubes)

    # Loop over order to draw cubes and draw them in image
    for cube in order_to_draw:
        create_cube(cube.position, cube.rotation, cube.colour, image_pil)

    # Gamma correction:
    image_np = np.array(image_pil)
    image_pil = (np.sqrt(image_np / 255.0) * 255).astype(np.uint8)

    # Convert the result back to PIL
    image_pil = Image.fromarray(image_pil, 'RGB')

    # Enable this line to save image
    # image_pil.save('image.png')
    # Show image
    image_pil.show()
