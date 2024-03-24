'''
Project By: Noga Levy and Tali Kalev
Lesson: Advanced Python
Date: March 31, 2024
'''

import numpy as np
from math import *
from PIL import Image, ImageDraw
from collections import namedtuple

# ====================MACROS====================
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
XYZ_VALUES = namedtuple("XYZ_VALUES", ["x", "y", "z"], defaults=(0,))
IMAGE_CENTER = [IMAGE_WIDTH/2, IMAGE_HEIGHT/2]
SCALE = 350

# ====================GIVEN DATA====================
CAMERA_POS   = XYZ_VALUES(0.736, -0.585, 0.338)
LIGHT_SOURCE = XYZ_VALUES(0.563, 0.139, 0.815)
BACKGROUND_COLOUR = np.array([0.5, 0.5, 0.5]) * 255 # PIL need 0-255 values and not 0-1 rgb values # RGB
# BACKGROUND_COLOUR = np.array([170, 170, 170])  # PIL need 0-255 values and not 0-1 rgb values # RGB

class CubeData():
    '''
    This Class contains all the data pertaining to a cube in the image including: position, rotation and colour
    '''
    def __init__(self, name, position, rotation, colour):
        self.name = name
        self.position = position
        self.rotation = rotation
        self.colour = colour

    def __str__(self):
        return f"{self.name}: Colour {self.colour}"

CUBE1 = CubeData(name="CUBE1", position=XYZ_VALUES(1.58, 0.08, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-57.7)), colour=np.array([0.6, 0.6, 0.6]) * 255) # GREY
CUBE2 = CubeData(name="CUBE2", position=XYZ_VALUES(0.58, -1.2, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-55.9)), colour=np.array([0.75, 0.5, 0.2]) * 255) # ORANGE
CUBE3 = CubeData(name="CUBE3", position=XYZ_VALUES(-0.75, 0.43, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-56.5)), colour=np.array([0.7, 0.3, 0.7]) * 255) # PURPLE
CUBE4 = CubeData(name="CUBE4", position=XYZ_VALUES(0.67, 1.15, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(0)), colour=np.array([0.47, 0.8, 0.42]) * 255) # GREEN


# CUBE1 = CubeData(name="CUBE1", position=XYZ_VALUES(1.58, 0.08, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-57.7)), colour=np.array([178, 178, 178])) # GREY
# CUBE2 = CubeData(name="CUBE2", position=XYZ_VALUES(0.58, -1.2, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-55.9)), colour=np.array([188, 171, 128])) # ORANGE
# CUBE3 = CubeData(name="CUBE3", position=XYZ_VALUES(-0.75, 0.43, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-56.5)), colour=np.array([185, 147, 186])) # PURPLE
# CUBE4 = CubeData(name="CUBE4", position=XYZ_VALUES(0.67, 1.15, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(0)), colour=np.array([ 168, 191, 163])) # GREEN


# ====================MATRICES CONSTANTS====================
# Define the camera's view direction (Z' vector from CAMERA_XYZ matrix)
CAMERA_VIEW_DIRECTION = np.array([CAMERA_POS.x, CAMERA_POS.y, CAMERA_POS.z])

# This is used to convert a 3D point to a 2D point
PROJECTION_MATRIX = np.array([
    [1, 0, 0],
    [0, 1, 0],
    # [0, 0, 0] # not necessary
])

# calculated manually
CAMERA_XYZ = np.array([
    [0.621, 0.782, 0], # X'
    [0.265, -0.211, -0.941], # -Y' 
    [0.736, -0.585, 0.338], # Z'
])

# indexed corners of cubes
FACES = [
    [0, 1, 2, 3], # Bottom face
    [5, 6, 7, 4], # Top face
    [0, 1, 5, 4], # Front face
    [2, 3, 7, 6], # Back face
    [0, 3, 7, 4], # Left face
    [1, 2, 6, 5], # Right face
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
# functions to get the roation matrix by given angle for each axis

def get_rotation_mat_z(angle):
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

def create_cube(cube_center, rotation_angle, image_pil, cube_colour): 
    '''
    Main function to create cubes in 2D image given 3D position and rotation.
    Receives center position of cube, rotation angle and image on which we are projecting.
    '''
    # Initialize an array to hold the transformed 3D points
    transformed_points_3d = np.zeros_like(POINTS)
    # Initialize an array to hold the transformed points in 2D representation
    projected_points = np.zeros((len(POINTS),2))

    for i, point in enumerate(POINTS):
        # Rotate object using rotation matrix with z axis angle
        rotated3d = np.dot(get_rotation_mat_z(rotation_angle.z),point)
        # translate point relative to cube's position (center of cube)
        translated_point = rotated3d + np.array([cube_center.x, cube_center.y, cube_center.z])
        # translate points to view from camera angle
        transformed_points_3d[i] = translated_point

        # convert from 3d point to 2d point using camera angle's view point
        projected2d = np.dot(PROJECTION_MATRIX, np.dot(CAMERA_XYZ, translated_point))
        # scale the x and y coordinates and place relative to center of image
        x = int(projected2d[0] * SCALE) + IMAGE_CENTER[0]
        y = int(projected2d[1] * SCALE) + IMAGE_CENTER[1]

        # Save cube point in 2D representation
        projected_points[i] = [x, y]

    draw_faces_of_cube(transformed_points_3d, cube_center, projected_points, cube_colour, image_pil) 


def find_face_normal_vector(points, face_indices, cube_center):
    '''
    Receives transformed 3D points of cube, 4 indices representing points on a face and the center of the cube.
    Returns the normal vector of the face.
    '''
    # Retrieve relevant points according to indices
    p1, p2, p3, p4 = [points[i] for i in face_indices]
    # Find midpoint of the cube face
    midpoint = (p2 + p4) / 2
    # Calculate vector from middle of cube to cube face
    vector_p = midpoint - cube_center
    return vector_p / np.linalg.norm(vector_p) # return normalized vector

def draw_face(image_pil, face_points_2d, cube_colour, face_normal_vector):
    '''
    Receives: image to draw on, 4 2D points, cube colour
    Draws the face of the cube represented by the 4 points onto the image taking into account the light source
    to change the shading on the cube face accordingly.
    '''
    # Find light source's relative angle on object and change colour accordingly
    light = np.dot(face_normal_vector, LIGHT_SOURCE)        
    cube_colour = tuple(int(value * light) for value in cube_colour)

    # Draw face of cube using 2D points + calculated colour
    face_points = [(point[0], point[1]) for point in face_points_2d]
    draw = ImageDraw.Draw(image_pil)
    draw.polygon(face_points, fill=(cube_colour))

def draw_faces_of_cube(transformed_points_3d, cube_center, projected_points, cube_colour, image_pil):
    '''
    Using the 3D transformed points and the 2D points, for each face of the cube we decide if it faces
    the angle of the camera and if it does, draws it in our image
    '''
    for face_indices in FACES:
        face_normal_vector = find_face_normal_vector(transformed_points_3d,face_indices, cube_center)
        # Check if face is in the direction of camera view
        if (np.dot(face_normal_vector, CAMERA_VIEW_DIRECTION) > 0):
            face_points_2d = [projected_points[i] for i in face_indices]
            draw_face(image_pil, face_points_2d, cube_colour, face_normal_vector)

def distance_from_camera(camera_pos, cube_pos):
    '''
    Function to calculate the distance from the camera to a cube using the cube center point
    '''
    return np.sqrt((camera_pos.x - cube_pos.x) ** 2 + 
                   (camera_pos.y - cube_pos.y) ** 2 + 
                   (camera_pos.z - cube_pos.z) ** 2)

def draw_order(camera_pos, cubes):
    '''
    Function to determine the order in which the cubes are drawn
    Returns list of CubeData objects where the first item should be drawn first as it is the furthest away
    from the camera and the last object is the closets
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
    image = create_image(IMAGE_HEIGHT, IMAGE_WIDTH)
    image_pil = Image.fromarray(image, 'RGB')

    # Find the order in which to draw the cubes depending on distance from camera
    cubes = [CUBE1, CUBE2, CUBE3, CUBE4]
    order_to_draw = draw_order(CAMERA_POS, cubes)

    # Loop over order to draw cubes and draw them in image
    for cube in order_to_draw:
        print(cube)
        create_cube(cube.position, cube.rotation, image_pil, cube.colour)

    # Save and show image
    image_pil.save('image2.png')
    image_pil.show()