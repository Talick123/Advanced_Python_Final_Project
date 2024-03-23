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

class CubeData():
    def __init__(self, name, position, rotation, colour):
        self.name = name
        self.position = position
        self.rotation = rotation
        self.colour = colour

    def __str__(self):
        return f"{self.name}: Colour {self.colour}"

CUBE1 = CubeData(name="CUBE1", position=XYZ_VALUES(1.58, 0.08, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-57.7)), colour=(153, 153, 153)) # GREY
CUBE2 = CubeData(name="CUBE2", position=XYZ_VALUES(0.58, -1.2, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-55.9)), colour=(191, 128, 51)) # ORANGE
CUBE3 = CubeData(name="CUBE3", position=XYZ_VALUES(-0.75, 0.43, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(-56.5)), colour=(179, 77, 179)) # PURPLE
CUBE4 = CubeData(name="CUBE4", position=XYZ_VALUES(0.67, 1.15, 0.5), rotation=XYZ_VALUES(0, 0, np.deg2rad(0)), colour=(120, 204, 107)) # GREEN

# ====================MATRICES CONSTANTS====================

# This is used to convert a 3D point to a 2D point
projection_matrix = np.array([
    [1, 0, 0],
    [0, 1, 0],
    # [0, 0, 0] # not necessary
])

# calculated manually
camera_xyz = np.array([
    [0.621, 0.782, 0], #X'
    [-0.265, 0.211, 0.941], #Y'
    [0.736, -0.585, 0.338], #Z'
])

# Define the camera's view direction (Z' vector from camera_xyz matrix)
camera_view_direction = np.array([CAMERA_POS.x, CAMERA_POS.y, CAMERA_POS.z])

# indexed corners of cubes
# finding normal vector to face via p1, p3, p4 : p4 - p1, p3 - p1
faces = [
    [0, 1, 2, 3], # Bottom face
    [5, 6, 7, 4], # Top face
    [0, 1, 5, 4], # Front face
    [2, 3, 7, 6], # Back face
    [0, 3, 7, 4], # Left face
    [1, 2, 6, 5], # Right face
]

points = np.array([[-0.5, -0.5, -0.5],  # index 0
                   [0.5, -0.5, -0.5],   # index 1
                   [0.5, 0.5, -0.5],    # index 2
                   [-0.5, 0.5, -0.5],   # index 3
                   [-0.5, -0.5, 0.5],   # index 4
                   [0.5, -0.5, 0.5],    # index 5
                   [0.5, 0.5, 0.5],     # index 6
                   [-0.5, 0.5, 0.5]])   # index 7

projected_points = np.zeros((len(points),2))

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

# ====================IMAGE CREATION FUNCTIONS==================== 
def create_image(height, width):
    '''
    Receives height and width of image
    Returns image with background colour represented as a 2D matrix
    '''
    bg_color = np.array([0.5, 0.5, 0.5]) * 255 # PIL need 0-255 values and not 0-1 rgb values
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[0:height, 0:width] = bg_color
    return image

def calculate_normal_of_cube_face(p1, p2, p3, p4):
    '''
    Function to calculate the normal vector of a face given four corner points
    '''
    # Calculate two vectors from the points of the face
    v1 = p4 - p1
    v2 = p3 - p1
    # Calculate the normal vector by taking the cross product of v1 and v2
    normal = np.cross(v1, v2)
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    return normal

def should_draw_face(points, face_indices):
    '''
    Function to determine if a face should be drawn by calculating the dot product the vector representing
    the normal of the face of the cube and the vector representing the camera angle
    '''
    # Get the corner points of the face
    p1, p2, p3, p4 = [points[i] for i in face_indices]
    # Calculate the normal vector of the face
    normal = calculate_normal_of_cube_face(p1, p2, p3, p4)
    # Calculate the dot product of the normal vector and the camera's view direction
    dot_product = np.dot(normal, camera_view_direction)
    # If the dot product is negative, the face is visible and should be drawn
    return dot_product < 0

def draw_face(image_pil, face_points_2d, cube_colour):
    '''
    Receives: image to draw on, 4 2D points, cube colour
    Draws the face of the cube represented by the 4 points onto the image
    '''
    face_points = [(point[0], point[1]) for point in face_points_2d]
    draw = ImageDraw.Draw(image_pil)
    draw.polygon(face_points, fill=(cube_colour))

def create_cube(cube_center, rotation_angle, image_pil, cube_colour): 
    '''
    Main function to create cubes in 2D image given 3D position and rotation.
    Receives center position of cube, rotation angle and image on which we are projecting.
    '''
    rotation_x = get_rotation_mat_x(rotation_angle.x)
    rotation_y = get_rotation_mat_y(rotation_angle.y)
    rotation_z = get_rotation_mat_z(rotation_angle.z)

    # Initialize an array to hold the transformed 3D points
    transformed_points_3d = np.zeros_like(points)

    for i, point in enumerate(points):
        # rotate object
        rotated3d = np.dot(rotation_z, np.dot(rotation_y, np.dot(rotation_x, point)))
        # translate point relative to cube's position (center of cube)
        translated_point = rotated3d + np.array([cube_center.x, cube_center.y, cube_center.z])
        # translate points to view from camera angle
        transformed_points_3d[i] = np.dot(camera_xyz, translated_point)

        # convert from 3d point to 2d point
        projected2d = np.dot(projection_matrix, transformed_points_3d[i])
        # scale the x and y coordinates and place relative to center of image
        x = int(projected2d[0] * SCALE) + IMAGE_CENTER[0]
        y = int(projected2d[1] * SCALE) + IMAGE_CENTER[1]

        projected_points[i] = [x, y]

    # Check if each face should be drawn
    for face_indices in faces:
        if should_draw_face(transformed_points_3d, face_indices):
            print(f"Face {face_indices} should be drawn.")
            face_points_2d = [projected_points[i] for i in face_indices]
            draw_face(image_pil, face_points_2d, cube_colour)
        else:
            print(f"Face {face_indices} should not be drawn.")

def distance_from_camera(camera_pos, cube_pos):
    '''
    Function to calculate the distance from the camera to a cube
    '''
    return np.sqrt((camera_pos.x - cube_pos.x) ** 2 + 
                   (camera_pos.y - cube_pos.y) ** 2 + 
                   (camera_pos.z - cube_pos.z) ** 2)

def draw_order(camera_pos, cubes):
    '''
    Function to determine the order in which the cubes are drawn
    '''
    # Calculate the distance from the camera to each cube and store it with the cube's identifier
    distances = [(cube, distance_from_camera(camera_pos, cube.position)) for cube in cubes]
    # Sort the cubes by their distance from the camera (furthest to closest)
    sorted_cubes = sorted(distances, key=lambda x: x[1], reverse=True)
    # Return the order in which the cubes should be drawn
    return [cube[0] for cube in sorted_cubes]


if __name__ == "__main__":
    # creating size + background
    image = create_image(IMAGE_HEIGHT, IMAGE_WIDTH)
    image_pil = Image.fromarray(image, 'RGB')

    cubes = [CUBE1, CUBE2, CUBE3, CUBE4]
    order_to_draw = draw_order(CAMERA_POS, cubes)

    print(f"The order in which the cubes should be drawn is:")
    for cube in order_to_draw:
        # if cube.name != "CUBE4":
        #     continue
        print(cube)
        create_cube(cube.position, cube.rotation, image_pil, cube.colour)

    image_pil.save('image2.png')
    image_pil.show()



# ======================================EXTRA======================================
    
# USE THIS TO DRAW 4 FACES OF A CUBE ALL IN DIFFERENT COLOURS IN create_cube

    # # NOGA: fill the area of 4 points, also added to the function "image_pil" argument
    # # # Fill the quadrilateral area with a specific color
    # quadrilateral_points = [tuple(point) for point in projected_points[:4]]
    # draw = ImageDraw.Draw(image_pil)
    # draw.polygon(quadrilateral_points, fill=(180, 20, 15))  # Fill with red color

    # quadrilateral_points = [tuple(point) for point in projected_points[4:]]
    # draw = ImageDraw.Draw(image_pil)
    # draw.polygon(quadrilateral_points, fill=(180, 20, 15))  # Fill with red color

    # # NOTE: the order of the points in the array matter!!!
    # y = [projected_points[1], projected_points[0], projected_points[4], projected_points[5]]
    # quadrilateral_points = [tuple(point) for point in y]
    # draw = ImageDraw.Draw(image_pil)
    # draw.polygon(quadrilateral_points, fill=(12, 200, 15))  # Fill with red color

    # # NOTE: the order of the points in the array matter!!!
    # y = [projected_points[3], projected_points[2], projected_points[6], projected_points[7]]
    # quadrilateral_points = [tuple(point) for point in y]
    # draw = ImageDraw.Draw(image_pil)
    # draw.polygon(quadrilateral_points, fill=(180, 200, 15))  # Fill with red color




# USE THIS TO DRAW 4 FACES OF A CUBE USING THE CUBES ACTUAL COLOUR
    # # # Fill the quadrilateral area with a specific color
    # quadrilateral_points = [tuple(point) for point in projected_points[:4]]
    # draw = ImageDraw.Draw(image_pil)
    # draw.polygon(quadrilateral_points, fill=(cube_colour))  

    # quadrilateral_points = [tuple(point) for point in projected_points[4:]]
    # draw = ImageDraw.Draw(image_pil)
    # draw.polygon(quadrilateral_points, fill=(cube_colour)) 

    # # NOTE: the order of the points in the array matter!!!
    # y = [projected_points[1], projected_points[0], projected_points[4], projected_points[5]]
    # quadrilateral_points = [tuple(point) for point in y]
    # draw = ImageDraw.Draw(image_pil)
    # draw.polygon(quadrilateral_points, fill=(cube_colour))  

    # # NOTE: the order of the points in the array matter!!!
    # y = [projected_points[3], projected_points[2], projected_points[6], projected_points[7]]
    # quadrilateral_points = [tuple(point) for point in y]
    # draw = ImageDraw.Draw(image_pil)
    # draw.polygon(quadrilateral_points, fill=(cube_colour))  #
