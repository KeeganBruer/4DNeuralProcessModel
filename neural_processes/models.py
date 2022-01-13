import torch
from torch import nn
from torch.nn import functional as F
from ray_tracing.ray_tracing import Ray, Sphere, get_closest_intersection_distance




class Encoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, x_dim)

        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """
    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class Decoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    TODO: replace the decoder neural network with a python function that assumes 
    z describes a set of spheres moving in the space that the Kinect rays intersect with. 
    sphere: [x, y, z, v1, v2, v3]
    Format: [sphere1, sphere2, ...]
    Result: [sph1_x, sph1_y, sph1_z, sph1_v1, sph1_v2, sph1_v3, 
            sph2_x, sph2_y, sph2_z, sph2_v1, sph2_v2, sph2_v3]

    X then becomes the source point, ray direction and time 
    Y is the closest intersection point of the ray and all the spheres.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.
    """
    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super(Decoder, self).__init__()
        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.z_sphere_dim = 6 #Defines the dimension of a single sphere
        self.sphere_radius = 1 #assume a set sphere dimension
        self.number_of_spheres = self.z_dim / self.z_sphere_dim
        self.distance_error = 0.002 #Kinect Camera Error


    def forward(self, x, z):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        Mu:
        Sigma:
        Sigma is the standard deviation of the point, ie. how much error is in it.  
        In this case, this represents the observation or measurement error. 
            if it was true that the real Kinects are observing actual spheres 
            flying through space, and X EXACTLY described these spheres, 
            how much error is left in the answer?  
        The Kinect has an observation error of less than 0.002 meters. 
        We can use this as a constant standard deviation for any point produced by the decoder, 
        just translate that distance into the dataset's normalized space.  
        """
        batch_size, num_points, _ = x.size()
        print(batch_size, num_points)
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        mu_arr = []
        sigma_arr = []
        for i in range(0, len(x_flat)): #For Each Ray
            x_in = x_flat[i]
            z_in = z_flat[i]
            ray = Ray(x_in[0:3], x_in[3], x_in[4:7])
            closest_distance = None #Closest distance per ray
            for j in range(0, len(z_in), 6): #Find closest intersection on every sphere
                sph_center = z_in[j:j+3] #first three
                sph_vec = z_in[j+3:j+6] #last three
                sphere = Sphere(sph_center, self.sphere_radius, sph_vec).play_sphere_forward(ray.time)
                dist = get_closest_intersection_distance(ray, sphere)
                if closest_distance == None or dist < closest_distance:
                    closest_distance = dist
            mu_arr.append(closest_distance.item())
            sigma_arr.append(self.distance_error)
        mu_tensor = torch.Tensor(mu_arr)
        sigma_tensor = torch.Tensor(sigma_arr)
        mu = mu_tensor.reshape(1, num_points, 1)
        sigma = sigma_tensor.reshape(1, num_points, 1)
        return mu, sigma



if (__name__ == "__main__"):
    decoder = Decoder(7, 6, 0, 1)
    x = torch.Tensor([[[0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0]]])
    z = torch.tensor([[5, 0, 0, 0, 0, 0]])
    res = decoder.forward(x, z)
    print(res)
