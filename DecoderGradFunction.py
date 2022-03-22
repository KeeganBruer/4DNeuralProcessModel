import torch
from neural_processes.ray_tracing.ray_tracing import Ray, Sphere, get_closest_intersection_distance
from SimulatedAnnealing import SimulatedAnnealing
params = None
sphere_radius = 1
distance_error = 0.002
def compute_mu_and_sigma(x, z):
    batch_size, num_points, x_dim = x.size()
    zbatch_size, z_dim = z.size()
    # Repeat z, so it can be concatenated with every x. This changes shape
    # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
    #print(z_in)

    # Flatten x and z to fit with linear layer
    x_flat = x.view(batch_size * num_points, x_dim)
    z_in = z.view(batch_size, z_dim)

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
    return mu_arr, sigma_arr


sim_an = SimulatedAnnealing()
def cost_func(state):
    x = state["x"]
    z = state["z"]
    mu_arr, sigma_arr = compute_mu_and_sigma(x, z)
    return 0
sim_an.get_cost = cost_func
def neighbors_func(state):
    new_state = {}
    new_state["x"] = state.x
    new_state["z"] = numpy.array([[0, 0, 2, 0, 0.1, 0, 4, 0, 5, 0, -0.1, -1]])
    return new_state

sim_an.get_neighbors = neighbors_func

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
        
        
        mu_arr, sigma_arr = compute_mu_and_sigma(x, z)
        mu_tensor = torch.tensor(mu_arr).float().to(z.device)
        sigma_tensor = torch.tensor(sigma_arr).float().to(z.device)

        mu = mu_tensor.reshape(1, num_points, 1)
        sigma = sigma_tensor.reshape(1, num_points, 1)
        mu_sigma = {
            "mu":mu,
            "sigma":sigma
        }
        ctx.save_for_backward(z, x, mu_sigma)
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
        learned_z, x, mu_sigma = ctx.saved_tensors
        initial_state = {
            "x": x,
            "z":learned_z,
            "target"
        }
        target_z = torch.tensor(
            sim_an.anneal(initial_state) #anneal numpy version of tensor
        ).float().to(learned_z.device)
        
        print(target_z)
        #target_z = torch.tensor([[0, 0, 2, 0, 0.1, 0, 4, 0, 5, 0, -0.1, -1]]).float().to(learned_z.device)


        
        z_error = torch.sub(learned_z, target_z)

        print(target_z)
        print(learned_z)
        print(z_error)
        learned_z.retain_grad()
        with torch.enable_grad(): 
                learned_z.backward(z_error,retain_graph=True)
        print(learned_z.grad)
        
        #grad = torch.autograd.grad(outputs=target_z, grad_outputs=z_error, inputs=learned_z)
        #print(grad)
        return (None, learned_z.grad)

        """
        return (
        	torch.autograd.Variable(), #X input requires no gradient
        	torch.autograd.Variable(torch.tensor([z_error]).type(torch.FloatTensor).to(learned_z.device), requires_grad=True)
        )
        """
        
        
        
        
