import numpy as np
from math import *
from PIL import Image

def draw_point(imagae, center, radius, color = [0.1, 0.1, 0.1]):
    color = np.array(color) * 255
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    # calc where to color the circle
    circle_mask = dist_from_center <= radius
    image[circle_mask] = color
    return image

# draw line from two given points in the matrix, i and j
def connect_points(image, i, j, points):
    color = [0, 0, 0]  # Black color
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


# === Some sizes for calculation:
h, w = 1080, 1920
scale = 100
circle_pos = [w/2, h/2]
normal = np.matrix([0, 0, 1])
camera = np.matrix([0.736, -0.585, 0.338]).reshape(3,1)
n_c_mul = np.dot(normal, camera)
angle = np.arccos(n_c_mul.flat[0])


angle = pi/4 # 36°
angleX = angle #np.deg2rad(-20)
angleY = angle #np.deg2rad(-57.7)
angleZ = angle #np.deg2rad(0) 

# === Some matrices for calculation:
projection_matrix = np.matrix([
    [1, 0, 0],
    [0, 1, 0]
    # [0, 0, 0] # not necessary
])

rotation_x = np.matrix([
    [1, 0, 0],
    [0, cos(angleX), -sin(angleX)],
    [0, sin(angleX), cos(angleX)],
])
rotation_y = np.matrix([
    [cos(angleY), 0, sin(angleY)],
    [0, 1, 0],
    [-sin(angleY), 0, cos(angleY)],
])
rotation_z = np.matrix([
    [cos(angleZ), -sin(angleZ), 0],
    [sin(angleZ), cos(angleZ), 0],
    [0, 0, 1],
])

# === cube points
# TODO: calculate the points by the given cube center
points = []

points.append(np.matrix([-1, -1, 1]))
points.append(np.matrix([1, -1, 1]))
points.append(np.matrix([1,  1, 1]))
points.append(np.matrix([-1, 1, 1]))
points.append(np.matrix([-1, -1, -1]))
points.append(np.matrix([1, -1, -1]))
points.append(np.matrix([1, 1, -1]))
points.append(np.matrix([-1, 1, -1]))

projected_points = [
    [n, n] for n in range(len(points))
]

# === bg
bg_color = np.array([0.5, 0.5, 0.5]) * 255 # PIL need 0-255 values and not 0-1 rgb values
image = np.zeros((h, w, 3), dtype=np.uint8)
image[0:h, 0:w] = bg_color

# === cube
i = 0
for point in points:
    # reshape so the points look like that: [[-1], [1], [1]] .... 
    # we actually want to rotate it only in x and y axis so we need to delete one of the rows here
    rotated2d = np.dot(rotation_z, point.reshape((3, 1)))
    rotated2d = np.dot(rotation_y, rotated2d) # NOGA:
    rotated2d = np.dot(rotation_x, rotated2d) # or this?


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
    
# some point
img = Image.fromarray(image, 'RGB')
img.save('my.png')
img.show()
















# ==================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# import numba as nb

# # @nb.njit()
# def create_circle(height, width):
#     # Create background
#     bg_color = np.array([0.5, 0.5, 0.5])
#     image = np.full((height, width, 3), bg_color)

#     # Create circle
#     circle_center = np.array([600, 400])
#     cube_side = 100
#     circle_color = np.array([0.8, 0.3, 0.1])  # Green color
    
#     Y, X = np.ogrid[:height, :width]
#     # dist_from_center = np.sqrt((X - circle_center[0])**2 + (Y - circle_center[1])**2)
#     # # calc where to color the circle
#     # circle_mask = dist_from_center <= circle_radius
#     cube_mask = in_cube(X, Y);
#     image[circle_mask] = circle_color

#     return image

# def in_cube(x, y):
#     if(x)


# # Generate circle
# circle_image = create_circle(1080, 1920)

# # Display circle
# plt.imshow(circle_image)
# plt.axis('off')
# plt.show()

# ============================== WITHOUT this matplotlib ==============================================
# from PIL import Image
# import numpy as np

# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
# img = Image.fromarray(data, 'RGB')
# img.save('my.png')
# img.show()


