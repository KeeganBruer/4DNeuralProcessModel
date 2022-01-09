import math
class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __add__(self, vector):
        x = self.x + vector.x
        y = self.y + vector.y
        z = self.z + vector.z
        return Vector3(x, y, z)
    def __sub__(self, vector):
        x = self.x - vector.x
        y = self.y - vector.y
        z = self.z - vector.z
        return Vector3(x, y, z)
    def __mul__(self, num):
        x = self.x * num
        y = self.y * num
        z = self.z * num
        return Vector3(x, y, z)
    def dot(self, vector):
        x = self.x * vector.x
        y = self.y * vector.y
        z = self.z * vector.z
        return x + y + z
    def length(self):
        sqrd_d = self.dot(self)
        return math.sqrt(sqrd_d)
    def __repr__(self):
        return [self.x, self.y, self.z]
    def __str__(self):
        return "{0} {1} {2}".format(self.x, self.y, self.z)

class Ray:
    def __init__(self, origin, time, direction):
        self.origin = Vector3(*origin)
        self.time = time
        self.direction = Vector3(*direction)
    def point_d_along(self, d):
        return self.origin + (self.direction * d)
    def __repr__(self):
        return [*self.origin, self.time, *self.direction]
    def __str__(self):
        return "origin:({0},{1},{2}) t:{3} dir:({4},{5},{6})".format(*self.origin, self.time, *self.direction)

class Sphere:
    def __init__(self, center, radius, velocity):
        self.center = Vector3(*center)
        self.radius = radius
        self.velocity = Vector3(*velocity)
    def __repr__(self):
        return [*self.origin, *self.velocity]
    def __str__(self):
        return "center:({0},{1},{2}) radius:{3} vel:({4},{5},{6})".format(*self.center, self.radius, *self.velocity)


def get_sphere_interections(ray, sphere):
    L = sphere.center - ray.origin
    print(L.length())
    tc = L.dot(ray.direction)
    print(tc)
    if tc < 0: return None, None
    print((tc*tc))
    print((L.dot(L)))
    t = (tc*tc) - (L.dot(L))
    if (t < 0):
        t = (L.dot(L)) - (tc*tc)
    print(t)
    d = math.sqrt(t)
    if (d > sphere.radius): return None, None

    t1c = math.sqrt((sphere.radius**2)-(d**2))
    t1 = tc - t1c
    t2 = tc + t1c
    return t1, t2

def get_intersection(ray, sphere):
    t1, t2 = get_sphere_interections(ray, sphere)
    print(t1, t2)
    return

def get_intersection_distance(ray, sphere):
    t1, t2 = get_sphere_interections(ray, sphere)
    return t1





