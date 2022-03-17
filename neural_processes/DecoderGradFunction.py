import torch
from neural_processes.ray_tracing.ray_tracing import Ray, DistanceRay, Sphere, get_closest_intersection_distance
params = None
sphere_radius = 1
distance_error = 0.002


class DecoderGradFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
  
    @staticmethod
    def forward(ctx, x, z):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        print("Custom Forward")
        z.retain_grad()
        
        

        batch_size, num_points, x_dim = x.size()
        #print(x.size())
        z_in = z[0]
        
        #print(z_in)
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
        
        error = Sphere([0, 0, 2], 1, [0, 0.1, 0]).get_error_from(spheres_from_z[0])
        error2 = Sphere([4, 0, 5], 1, [0, -0.1, -1]).get_error_from(spheres_from_z[1])
        #print("Distance from learned Z to target Z center - 1st   {}".format(error["center_error"]))
        #print("Distance from learned Z to target Z velocity - 1st {}".format(error["velocity_error"]))
        #print("Distance from learned Z to target Z center - 2nd   {}".format(error["center_error_length"]))
        #print("Distance from learned Z to target Z velocity - 2nd {}".format(error["velocity_error_length"]))
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
        mu_tensor = torch.tensor(mu_arr, dtype=torch.float32, requires_grad=True).to(z.device)
        sigma_tensor = torch.tensor(sigma_arr, requires_grad=False).float().to(z.device)

        mu = mu_tensor.reshape(1, num_points, 1)

        sigma = sigma_tensor.reshape(1, num_points, 1)
        ctx.save_for_backward(x, z, mu)

        return mu, sigma

    @staticmethod
    def backward(ctx, mu_grad_output, sigma_grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        print("Custom Backwards")
        #print(mu_grad_output, sigma_grad_output)
        x, learned_z, out_mu = ctx.saved_tensors
        batch_size, num_points, x_dim = x.size()
        _, z_dim = learned_z.size()
        z_flat = learned_z.view(batch_size * z_dim)
        print(len(z_flat)/6)
        spheres_from_z = []
        for j in range(0, len(z_flat), 6): #Find closest intersection on every sphere
                sph_center = z_flat[j:j+3] #first three
                sph_vec = z_flat[j+3:j+6] #last three
                #print(sph_center)
                sphere = Sphere(sph_center, sphere_radius, sph_vec)
                spheres_from_z.append(sphere)
        
        x_flat = x.view(batch_size * num_points, x_dim)
        mu_flat = out_mu.view(batch_size * num_points, 1)
        rays = []
        for i in range(0, len(x_flat)): #For Each Ray
            x_in = x_flat[i].tolist()
            dist = mu_flat[i].item()
            ray = DistanceRay(x_in[0:3], x_in[3], x_in[4:7], dist)
            rays.append(ray)
        
        _, _, mu_dim = mu_grad_output.size()
        error_from_grad = mu_grad_output.view(batch_size * num_points, mu_dim)
        target_z = error2targetz(rays, spheres_from_z, error_from_grad)
        #target_z = torch.tensor([[0, 0, 2, 0, 0.1, 0, 4, 0, 5, 0, -0.1, -1]]).float().to(learned_z.device)
        
        z_error = torch.sub(learned_z, target_z)

        #print(target_z)
        #print(learned_z)
        #print(z_error)
        learned_z.retain_grad()
        with torch.enable_grad(): 
                learned_z.backward(z_error,retain_graph=True)
        #print(learned_z.grad)
        
        #grad = torch.autograd.grad(outputs=target_z, grad_outputs=z_error, inputs=learned_z)
        #print(grad)
        return (None, learned_z.grad)

        """
        return (
        	torch.autograd.Variable(), #X input requires no gradient
        	torch.autograd.Variable(torch.tensor([z_error]).type(torch.FloatTensor).to(learned_z.device), requires_grad=True)
        )
        """
        
        
def error2targetz(rays, curr_spheres, error):
        print(len(rays), len(curr_spheres), len(error))
        for i in range(0, len(rays)):
        	dray = rays[i]
        	err = error[i].item()
        	print(dray.distance, err)
        	current_contact = Ray(distance_ray=dray).point_d_along(dray.distance)
        	target_contact = Ray(distance_ray=dray).point_d_along(dray.distance + err)
        	for sphere in curr_spheres:
        		time_progressed_sphere = sphere.play_sphere_forward(dray.time)
        		print((target_contact-sphere.center).length())
        	print(i, current_contact)
        	print(target_contact, "\n")
        return torch.tensor([[0, 0, 2, 0, 0.1, 0, 4, 0, 5, 0, -0.1, -1]]).float().to(error.device)
        	
       
