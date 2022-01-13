import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
from ray_tracing import Ray, Sphere, get_closest_intersection_distance


width = 100
height = 100
ppi = 30
rays = []
spheres = []
for x in range(-1*int(width/2), int(width/2)):
    for y in range(-1*int(height/2), int(height/2)):
        new_x = x / ppi
        new_y = y / ppi
        rays.append(Ray([0, 0, 0], 0, [new_x, new_y, 1]).unit())
#print(rays)   
spheres.append(Sphere([0, 0, 2], 1, [0, 0.1, 0]))
spheres.append(Sphere([4, 0, 5], 1, [0,-0.1,-1]))
fig, ax = plt.subplots()


def update(time):
    print("time", time)
    largest_distance = 10
    img_data = np.zeros( (width, height, 3), dtype=np.uint8)
    for i in range(0, len(rays)):
        x = i % width
        y = int(i / width)
        ray = rays[i]
        closest_dist = None
        for j in range(0, len(spheres)):
            sphere = spheres[j].play_sphere_forward(time)
            dist = get_closest_intersection_distance(ray, sphere)
            if dist != None and (closest_dist == None or dist < closest_dist):
                closest_dist = dist
        if closest_dist != None:
            perc = closest_dist/largest_distance
            dist = perc * (255*3)
            r, g, b = 0, 0, 0
            if dist > 255:
                r = 255
                dist -= 255
            else:
                r = dist
                dist = 0
            if dist > 255:
                g = 255
                dist -= 255
            else:
                g = dist
                dist = 0
            if dist > 255:
                b = 255
                dist -= 255
            else:
                b = dist
                dist = 0
            
            img_data[x, y] = [r, g, b]
        else:
            img_data[x, y] = [0, 0, 0]
    ax.imshow(img_data)
    return ax

fps = 2
anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=1000/fps)
anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
