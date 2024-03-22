import numpy as np
from math import *
from PIL import Image

def draw_point(image, center, radius, color = [0.1, 0.1, 0.1]):
    color = np.array(color) * 255
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    # calc where to color the circle
    circle_mask = dist_from_center <= radius
    image[circle_mask] = color
    return image

# draw line from two given points in the matrix, i and j
def connect_points(image, i, j, points):
    print(f"i: {i}, j:{j}")
    color = [0, 0, 0]  # Black color
    if i == 0 and j == 1:
        print("HELLO")
        color = [1.,1,0]
    x1, y1 = points[i][0], points[i][1]
    x2, y2 = points[j][0], points[j][1]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    # Initialize the error variable
    error = dx - dy

    while x1 != x2 or y1 != y2:
        # Draw the current point
        image[int(y1), int(x1)] = color

        # Calculate the next point
        error2 = 2 * error
        if error2 > -dy:
            error -= dy
            x1 += sx
        if error2 < dx:
            error += dx
            y1 += sy
    
    return image

# create the full image with the background and size
def create_image(h, w):
    # === bg
    bg_color = np.array([0.5, 0.5, 0.5]) * 255 # PIL need 0-255 values and not 0-1 rgb values
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[0:h, 0:w] = bg_color
    return image
    pass

# some of the variables here can be global constants (I think) - depending on whether we need them or not
# for example : projection_matrix, rotation_z/y/z, scale, projected_points,
# points (?) also depending on how we calculate it
def create_cube(image, circle_pos, angleX, angleY, angleZ): 
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
        x = int(projected2d.flat[0] * scale) + circle_pos[0]
        y = int(projected2d.flat[1] * scale) + circle_pos[1]

        projected_points[i] = [x, y]

        # delete later
        if i == 0: color = [1,0,0]
        elif i == 1: color = [0,0,1] 
        elif i == 2: color = [1,1,0] 
        else: color = [0,0,0]

        image = draw_point(image, np.array([x, y]), 5, color)
        i += 1

    for p in range(4):
        connect_points(image, p, (p+1) % 4, projected_points)
        connect_points(image, p+4, ((p+1) % 4) + 4, projected_points)
        connect_points(image, p, (p+4), projected_points)

    return image

# === Some matrices for calculation:
# also functions to get the roation matrix by given angle for each axis
def get_rotation_mat_x(angle):
    return np.matrix([
        [1, 0, 0],
        [0, cos(angle), -sin(angle)],
        [0, sin(angle), cos(angle)],
    ])

def get_rotation_mat_y(angle):
    return np.matrix([
        [cos(angle), 0, sin(angle)],
        [0, 1, 0],
        [-sin(angle), 0, cos(angle)],
    ])

def get_rotation_mat_z(angle):
    return np.matrix([
        [cos(angle), -sin(angle), 0],
        [sin(angle), cos(angle), 0],
        [0, 0, 1],
    ])

# like constant maybe
projection_matrix = np.matrix([
    [1, 0, 0],
    [0, 1, 0]
    # [0, 0, 0] # not necessary
])

# def camera():
#     # the z' perpendicular to y' and x' to z and y'
#     camera_z = [0.736, -0.585, 0.338]
#     camera_y = [-0.736, 0.585]
#     c = (camera_z[0] * camera_y[0] + camera_z[1] * camera_y[1]) / (-0.338)
#     camera_y.append(c)
#     magnitude = np.linalg.norm(camera_y)
#     # Normalize the vector
#     normalized_vector = camera_y / magnitude
#     print(normalized_vector ,camera_z)


# camera()
camera_xyz = np.matrix([
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

# === Some sizes for calculation:
h, w = 1080, 1920 # like constant maybe
scale = 100  # like constant maybe
circle_pos1 = [w/2, h/2]
circle_pos2 = [w/4, h/3]
# normal = np.matrix([0, 0, 1])
# camera = np.matrix([0.736, -0.585, 0.338]).reshape(3,1)
# n_c_mul = np.dot(normal, camera)
# angle = np.arccos(n_c_mul.flat[0])
angle = pi/4 # 36°
angleX = angle #np.deg2rad(-20)
angleY = angle #np.deg2rad(-57.7)
angleZ = angle #np.deg2rad(0) 

# size + bg
image = create_image(1080, 1920)

# cube1
image = create_cube(image, circle_pos1, 0, 0, np.deg2rad(-57.7))
# cube2
image = create_cube(image, circle_pos2, 0, 0, np.deg2rad(-55.9)) # OK, cool, 

# display + save as png image
img = Image.fromarray(image, 'RGB')
img.save('my.png')
img.show()


#================================================================== from cubeFaces.py

def draw_point(image, center, radius, color = [0.1, 0.1, 0.1]):
    '''
    Temporary function for development
    Draws a point in an image represented by a 2D matrix with a given color
    '''
    color = np.array(color) * 255
    Y, X = np.ogrid[:IMAGE_HEIGHT, :IMAGE_WIDTH]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    # calc where to color the circle
    circle_mask = dist_from_center <= radius
    image[circle_mask] = color
    return image

def connect_points(image, i, j, points):
    '''
    Temporary function for development
    Draws a line in an image represented by a 2D matrix connecting two given points
    '''
    # print(f"i: {i}, j:{j}")
    color = [0, 0, 0]  # Black color
    if i == 0 and j == 1:
        # print("HELLO")
        color = [1.,1,0]
    x1, y1 = points[i][0], points[i][1]
    x2, y2 = points[j][0], points[j][1]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    # Initialize the error variable
    error = dx - dy

    while x1 != x2 or y1 != y2:
        # Draw the current point
        image[int(y1), int(x1)] = color

        # Calculate the next point
        error2 = 2 * error
        if error2 > -dy:
            error -= dy
            x1 += sx
        if error2 < dx:
            error += dx
            y1 += sy
    
    return image
