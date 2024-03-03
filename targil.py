import numpy as np
from math import *
from PIL import Image

def draw_point(imagae, center, radius):
    color = np.array([0., 0.2, 0.2]) * 255

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    # calc where to color the circle
    circle_mask = dist_from_center <= radius
    image[circle_mask] = color
    return image


# cube points
points = []

points.append(np.matrix([-1, -1, 1]))
points.append(np.matrix([1, -1, 1]))
points.append(np.matrix([1,  1, 1]))
points.append(np.matrix([-1, 1, 1]))
points.append(np.matrix([-1, -1, -1]))
points.append(np.matrix([1, -1, -1]))
points.append(np.matrix([1, 1, -1]))
points.append(np.matrix([-1, 1, -1]))

# bg
h, w = 1080, 1920

bg_color = np.array([0.5, 0.5, 0.5]) * 255 # PIL need 0-255 values and not 0-1 rgb values
image = np.zeros((h, w, 3), dtype=np.uint8)
image[0:h, 0:w] = bg_color

# some point
image = draw_point(image, np.array([600, 400]), 5)
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


