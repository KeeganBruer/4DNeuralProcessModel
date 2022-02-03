import math
class Vector3:
    def __init__(self, x, y, z):
        self._type_ = "Vector3"
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
    def unit(self):
        length = self.length()
        x = self.x / length
        y = self.y / length
        z = self.z / length
        return Vector3(x, y, z)
    def make_axis(self, x=None, y=None, z=None):
        if z != None:
            ratio = z / self.z
            new_x = self.x * ratio
            new_y = self.y * ratio
            return Vector3(new_x, new_y, z)
    def toArray(self):
        return [self.x, self.y, self.z]
    def __repr__(self):
        return "{0} {1} {2}".format(self.x, self.y, self.z)
    def __str__(self):
        return "{0} {1} {2}".format(self.x, self.y, self.z)

class Ray:
    def __init__(self, origin, time, direction):
        self._type_ = "Ray"
        self.origin = Vector3(*origin)
        self.time = time
        self.direction = Vector3(*direction)
    def point_d_along(self, d):
        return self.origin + (self.direction * d)
    def unit(self):
        """
        Returns a copy of the Ray with the direction vector in unit form.
        """
        return Ray(self.origin.toArray(), self.time, self.direction.unit().toArray())
    def __repr__(self):
        return str(self)
    def __str__(self):
         return (
            "origin:({0}) "+
            "t:{1} "+
            "dir:({2})"
        ).format(self.origin, self.time, self.direction)
class DistanceRay(Ray):
    def __init__(self, origin=None, time=None, direction=None, distance=0, ray=None):
        if ray != None:
            self.origin = ray.origin
            self.time = ray.time
            self.direction = ray.direction
        if origin != None:
            self.origin = origin
        elif (ray == None):
            self.origin = Vector3(0, 0, 0)
        if time != None:
            self.time = time
        elif (ray == None):
            self.time = 0
        if direction != None:
            self.direction = direction
        elif (ray == None):
            self.direction = Vector3(0, 0, 1)
        self.distance = distance
    def to_dataset_format(self):
        return [
            *self.origin.toArray(),
            self.time,
            *self.direction.toArray(),
        ], self.distance
    def from_dataset_format(self, x, y):
        self.origin = Vector3(*x[0:3])
        self.time = x[3]
        self.direction = Vector3(*x[4:7])
        self.distance = y
        return self
    def __repr__(self):
        return str(self)
    def __str__(self):
         return (
            "origin:({0}) "+
            "t:{1} "+
            "dir:({2}) "+
            "dist: {3}"
        ).format(self.origin, self.time, self.direction, self.distance)
        

class Sphere:
    def __init__(self, center, radius, velocity):
        self._type_ = "sphere"
        self.center = Vector3(*center)
        self.radius = radius
        self.velocity = Vector3(*velocity)
    def play_sphere_forward(self, t):
        center = self.center
        vel = self.velocity
        x = center.x + (vel.x * t)
        y = center.y + (vel.y * t)
        z = center.z + (vel.z * t)
        return Sphere([x, y, z], self.radius, vel.toArray())

    def __repr__(self):
        return [*self.origin, *self.velocity]
    def __str__(self):
        return "center:({0}) radius:{1} vel:({2})".format(self.center, self.radius, self.velocity)


def get_sphere_interections(ray, sphere):
    L = sphere.center - ray.origin

    tc = L.dot(ray.direction)

    if tc < 0: return None, None

    t = (tc*tc) - (L.dot(L))
    if (t < 0):
        t = (L.dot(L)) - (tc*tc)

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

def get_closest_intersection_distance(ray, sphere):
    t1, t2 = get_sphere_interections(ray, sphere)
    return t1





