import torch
from neural_processes.ray_tracing.ray_tracing import Ray, Sphere, get_closest_intersection_distance
params = None



class DecoderGradFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    params = None
    
    @classmethod
    def set_params(self, x_dim, sphere_radius, distance_error):
	    self.params = (x_dim, sphere_radius, distance_error)

    @classmethod
    def forward(self, ctx, x, z):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        x_dim, sphere_radius, distance_error = self.params
        print("Custom Forward")
        ctx.save_for_backward(z)
        batch_size, num_points, _ = x.size()
        z_in = z[0]
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        #print(z_in)

        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, x_dim)

        
        spheres_from_z = []
        for j in range(0, len(z_in), 6): #Find closest intersection on every sphere
                sph_center = z_in[j:j+3] #first three
                sph_vec = z_in[j+3:j+6] #last three
                #print(sph_center)
                sphere = Sphere(sph_center, sphere_radius, sph_vec)
                spheres_from_z.append(sphere)
        
        error = Sphere([0, 0, 1], 1, [0, 0.1, 0]).get_error_from(spheres_from_z[0])
        error2 = Sphere([4, 0, 5], 1, [0, -0.1, -1]).get_error_from(spheres_from_z[1])
        print("Distance from learned Z to target Z center - 1st   {}".format(error["center_error_length"]))
        print("Distance from learned Z to target Z velocity - 1st {}".format(error["velocity_error_length"]))
        print("Distance from learned Z to target Z center - 2nd   {}".format(error["center_error_length"]))
        print("Distance from learned Z to target Z velocity - 2nd {}".format(error["velocity_error_length"]))
        mu_arr = []
        sigma_arr = []
        for i in range(0, len(x_flat)): #For Each Ray
            x_in = x_flat[i].tolist()

            ray = Ray(x_in[0:3], x_in[3], x_in[4:7])
            closest_distance = None #Closest distance per ray
            for j in range(0, len(spheres_from_z)):
                sphere = spheres_from_z[j].play_sphere_forward(ray.time)
                dist = get_closest_intersection_distance(ray, sphere)
                if closest_distance == None or (dist != None and dist < closest_distance):
                    closest_distance = dist
            if (closest_distance != None):
                mu_arr.append(closest_distance)
            else:
                mu_arr.append(0)
            sigma_arr.append(distance_error)
        mu_tensor = torch.Tensor(mu_arr).to(z.device)
        sigma_tensor = torch.tensor(sigma_arr, requires_grad=True).to(z.device)

        mu = mu_tensor.reshape(1, num_points, 1)
        sigma = sigma_tensor.reshape(1, num_points, 1)

        return mu, sigma

    @classmethod
    def backward(self, ctx, mu_grad_output, sigma_grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        print("Custom Backwards")
        #print(mu_grad_output, sigma_grad_output)
        learned_z, = ctx.saved_tensors
        
        return (
            torch.autograd.Variable(torch.tensor([[10 for x in range(12)]]).type(torch.FloatTensor).to(learned_z.device), requires_grad=True), 
            torch.autograd.Variable(torch.tensor([[10 for x in range(12)]]).type(torch.FloatTensor).to(learned_z.device), requires_grad=True)
        )
