# 4DNeuralProcessModel
The model is formatted to match: <br>


| Model | [x1,y1,z1,t1,x2,y2,z2,t2] => [distance] | 
| --------------- | ----------- |
| x1,y1,z1,t1 | Origin of the ray  |
| x2,y2,z2,t2 | The max point along the ray (endpoint) |
| distance    | A floating point number representing the percentage of the distance between the origin and endpoint that is where the ray intersects with an object. |



# Neural Process Implementation Pulled From The Repository By [EmilienDupont](https://github.com/EmilienDupont/neural-processes)
The Neural Process has been adapted to use a custom decoder.

# Custom Encoder
We replaced the decoder neural network with a python function that assumes z describes a set of spheres moving in the space that the Kinect rays intersect with. X then becomes the source point, ray direction and time, then y is the closest intersection point of the ray and all the spheres.


