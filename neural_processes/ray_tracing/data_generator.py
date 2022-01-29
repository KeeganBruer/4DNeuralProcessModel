import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
from ray_tracing import Ray, DistanceRay, Sphere, get_closest_intersection_distance

total_frames = 30
width = 500
height = 500
ppi = 50
ray_max_dist = 10
rays = []
spheres = []
for x in range(-1*int(width/2), int(width/2)):
    for y in range(-1*int(height/2), int(height/2)):
        new_x = x / ppi
        new_y = y / ppi
        rays.append(Ray([0, 0, 0], 0, [new_x, new_y, 1]).unit())
#print(rays)   
spheres.append(Sphere([0, 0, 2], 1, [0, 0.1, 0]))
spheres.append(Sphere([4, 0, 5], 1, [0, -0.1, -1]))

ray_frames = []
curr_spheres = spheres[:]
for t in range(0, total_frames+1):
    ray_frame = []
    for i in range(0, len(rays)):
        ray = DistanceRay(ray=rays[i], time=t, distance=None)
        for j in range(0, len(spheres)):
            sphere = curr_spheres[j].play_sphere_forward(t)
            dist = get_closest_intersection_distance(ray, sphere)
            if dist != None and (ray.distance == None or dist < ray.distance):
                ray.distance = dist
        if (ray.distance == None):
            ray.distance = 0
        ray_frame.append(ray)
    ray_frames.append(ray_frame)


X = []
Y = []
for i in range(0, len(ray_frames)):
    ray_frame = ray_frames[i]
    for j in range(0, len(ray_frame)):
        x, y = ray_frame[j].to_dataset_format()
        X.append(x)
        Y.append(y)
np.savez_compressed(
    "dataset_out.npz", 
    X=X, Y=Y, 
    width=width, 
    height=height,
    total_frames=total_frames,
    ppi=ppi
)

"""
depth_data = np.zeros( (width, height, 1), dtype=np.uint8)
for i in range(0, len(rays)):
    x = i % width
    y = int(i / width)
    ray = rays[i]
    depth_data[x, y] = ray.distance 
fig, ax = plt.subplots()
color_map = plt.cm.get_cmap('gray')
reversed_color_map = color_map.reversed()
ax.imshow(depth_data, cmap=reversed_color_map, vmin=0, vmax=10)
plt.show()
"""