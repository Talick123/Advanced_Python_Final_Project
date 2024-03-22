import numpy as np
from math import *
from PIL import Image, ImageDraw
from collections import namedtuple

# ====================MACROS====================
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
XYZ_VALUES = namedtuple("XYZ_VALUES", ["x", "y", "z"])

# ====================GIVEN DATA====================
CAMERA_POS   = XYZ_VALUES(0.736, -0.585, 0.338)
LIGHT_SOURCE = XYZ_VALUES(0.563, 0.139, 0.815)
CUBE1_POS    = XYZ_VALUES(1.58, 0.08, 0.5)
CUBE2_POS    = XYZ_VALUES(0.58, -1.2, 0.5)
CUBE3_POS    = XYZ_VALUES(-0.75, 0.43, 0.5)
CUBE4_POS    = XYZ_VALUES(0.67, 1.15, 0.5)

CUBE1_ROT    = XYZ_VALUES(0, 0, np.deg2rad(-57.7))
CUBE2_ROT    = XYZ_VALUES(0, 0, np.deg2rad(-55.9))
CUBE3_ROT    = XYZ_VALUES(0, 0, np.deg2rad(-56.5))
CUBE4_ROT    = XYZ_VALUES(0, 0, 0)

# ====================MATRICES CONSTANTS====================
projection_matrix = np.array([
    [1, 0, 0],
    [0, 1, 0]
    # [0, 0, 0] # not necessary
])

# camera()
camera_xyz = np.array([
    [0.621, 0.782, 0], #X'
    [-0.265, 0.211, 0.941], #Y'
    [0.736, -0.585, 0.338], #Z'
])

# === cube points
# TODO: calculate the points by the given cube center
points = np.array([
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1]
])

projected_points = [
    [n, n] for n in range(len(points))
]

# ====================CALCULATION MATRICES====================
# functions to get the roation matrix by given angle for each axis

def get_rotation_mat_x(angle):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)],
    ])

def get_rotation_mat_y(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])

def get_rotation_mat_z(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])

# ====================TEMP CALCULATIONS==================== will most likely be removed
scale = 100
center_pos1 = [IMAGE_WIDTH/2, IMAGE_HEIGHT/2]
center_pos2 = [IMAGE_WIDTH/4, IMAGE_HEIGHT/3]

def create_image(height, width):
    '''
    Receives height and width of image
    Returns image with background colour represented as a 2D matrix
    '''
    bg_color = np.array([0.5, 0.5, 0.5]) * 255 # PIL need 0-255 values and not 0-1 rgb values
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[0:height, 0:width] = bg_color
    return image

# some of the variables here can be global constants (I think) - depending on whether we need them or not
# for example : projection_matrix, rotation_z/y/z, scale, projected_points,
# points (?) also depending on how we calculate it
def create_cube(center_pos, angleX, angleY, angleZ, image_pil): 
    '''
    Main function to create cubes in 2D image given 3D position and rotation
    '''
    rotation_x = get_rotation_mat_x(angleX)
    rotation_y = get_rotation_mat_y(angleY)
    rotation_z = get_rotation_mat_z(angleZ)
    
    i = 0
    for point in points:
        # reshape so the points look like that: [[-1], [1], [1]] .... 
        # we actually want to rotate it only in x and y axis so we need to delete one of the rows here
        rotated2d = np.dot(rotation_z, point.reshape((3, 1)))
        rotated2d = np.dot(rotation_y, rotated2d) # NOGA:
        rotated2d = np.dot(rotation_x, rotated2d) # or this?
        rotated2d = np.dot(camera_xyz, rotated2d) # or this?

        # convert from 3d point to 2d point
        projected2d = np.dot(projection_matrix, rotated2d)
        # scale the x and y coordinates
        x = int(projected2d.flat[0] * scale) + center_pos[0]
        y = int(projected2d.flat[1] * scale) + center_pos[1]

        projected_points[i] = [x, y]

        i += 1

    # NOGA: fill the area of 4 points, also added to the function "image_pil" argument
    # # Fill the quadrilateral area with a specific color
    quadrilateral_points = [tuple(point) for point in projected_points[:4]]
    draw = ImageDraw.Draw(image_pil)
    draw.polygon(quadrilateral_points, fill=(180, 20, 15))  # Fill with red color

    quadrilateral_points = [tuple(point) for point in projected_points[4:]]
    draw = ImageDraw.Draw(image_pil)
    draw.polygon(quadrilateral_points, fill=(180, 20, 15))  # Fill with red color

    # NOTE: the order of the points in the array matter!!!
    y = [projected_points[1], projected_points[0], projected_points[4], projected_points[5]]
    quadrilateral_points = [tuple(point) for point in y]
    draw = ImageDraw.Draw(image_pil)
    draw.polygon(quadrilateral_points, fill=(12, 200, 15))  # Fill with red color

    # NOTE: the order of the points in the array matter!!!
    y = [projected_points[3], projected_points[2], projected_points[6], projected_points[7]]
    quadrilateral_points = [tuple(point) for point in y]
    draw = ImageDraw.Draw(image_pil)
    draw.polygon(quadrilateral_points, fill=(180, 200, 15))  # Fill with red color


if __name__ == "__main__":
    # creating size + background
    image = create_image(IMAGE_HEIGHT, IMAGE_WIDTH)

    image_pil = Image.fromarray(image, 'RGB')

    #TODO convert 3D cube centers to 2D cube centers

    # cube1
    create_cube(center_pos1, 0, 0, CUBE1_ROT.z, image_pil)
    # cube2
    create_cube(center_pos2, 0, 0, CUBE4_ROT.z, image_pil)

    image_pil.save('image2.png')
    image_pil.show()

