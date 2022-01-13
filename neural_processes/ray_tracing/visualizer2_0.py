import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
from ray_tracing import Ray, DistanceRay, Sphere, get_closest_intersection_distance

class Visualize:
    def __init__(self, _type_="generate", file="", save_animation=False, show=True):
        self.width = 100
        self.height = 100
        self.ppi = 10
        self.ray_max_dist = 10
        self.max_time = 30
        self.fps = 5
        self.fig, self.ax = plt.subplots()
        if (_type_ == "generate"):
            self.generate_rays_and_spheres()
        elif (_type_ == "load"):
            self.import_rays_from_file(file)
        self.build_frames_frome_rays_and_spheres()
        anim = FuncAnimation(self.fig, self.animation_update, frames=np.arange(0, self.max_time), interval=1000/self.fps)
        if (save_animation):
            anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
        if (show):
            plt.show()

    def generate_rays_and_spheres(self):
        rays = []
        spheres = []
        for x in range(-1*int(self.width/2), int(self.width/2)):
            for y in range(-1*int(self.height/2), int(self.height/2)):
                new_x = x / self.ppi
                new_y = y / self.ppi
                rays.append(Ray([0, 0, 0], 0, [new_x, new_y, 1]).unit())
        #print(rays)   
        spheres.append(Sphere([0, 0, 2], 1, [0, 0.1, 0]))
        spheres.append(Sphere([0, 0, 5], 1, [0, -0.1, 0]))
        self.rays = rays
        self.spheres = spheres
    def import_rays_from_file(self, file):
        data = np.load(file)
        print(data.files)
        X = data['X']
        Y = data['Y']
        rays = []
        for i in range(0, len(X)):
            x = X[i]
            y = Y[i]
            rays.append(DistanceRay().from_dataset_format(x, y))
        print(rays)

    def build_frames_frome_rays_and_spheres(self):
        self.ray_frames = []
        curr_spheres = self.spheres
        print(len(curr_spheres))
        for t in range(0, self.max_time):
            ray_frame = []
            for i in range(0, len(self.rays)):
                ray = DistanceRay(ray=self.rays[i], time=t, distance=self.ray_max_dist)
                for j in range(0, len(curr_spheres)):
                    sphere = curr_spheres[j].play_sphere_forward(t)
                    dist = get_closest_intersection_distance(ray, sphere)
                    if dist != None and dist < ray.distance:
                        ray.distance = dist
                ray_frame.append(ray)
            self.ray_frames.append(ray_frame)
        print("Finished Building Frames")

    def animation_update(self, time):
        img_data = np.zeros( (self.width, self.height, 3), dtype=np.uint8)
        ray_frame = self.ray_frames[time]
        for i in range(0, len(ray_frame)):
            x = i % self.width
            y = int(i / self.width)
            ray = ray_frame[i]
            dist = ray.distance
            perc = dist/self.ray_max_dist

            color = perc * (255*3)

            r, g, b = 0, 0, 0
            if color > 255:
                r = 255
                color -= 255
            else:
                r = color
                color = 0
            if color > 255:
                g = 255
                color -= 255
            else:
                g = color
                color = 0
            if color > 255:
                b = 255
                color -= 255
            else:
                b = color
                color = 0

            img_data[x, y] = [r, g, b]
        self.ax.imshow(img_data)
        return self.ax


Visualize(_type_="load", file="dataset_out.npz")