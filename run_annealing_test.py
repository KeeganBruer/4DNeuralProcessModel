
import numpy
from DecoderGradFunction import compute_mu_and_sigma
def get_sample(count, *args):
    A = args[0]
    indexs = numpy.random.choice(A.shape[0], count, replace=False)
    print(indexs, len(A.shape))
    rtn = []
    for i in range(0, len(args)):
        rtn.append(args[i][indexs])
    return rtn
    
raw_data = numpy.load("./training_data/dataset_out.npz")
x = raw_data["X"]
y = raw_data["Y"]
#x, y = get_sample(20000, x, y)
print(x.shape)

target_x = [0, 0, 2, 0, 0.1, 0, 4, 0, 5, 0, -0.1, -1]
initial_z = numpy.random.rand(12)
print("Initial Z", initial_z)
#[0, 0, 1, 0, 0.02, 0, 2, 0, 5, 0, 0.4, 0]

from SimulatedAnnealing import SimulatedAnnealing
params = None
sphere_radius = 1
distance_error = 0.002
sim_an = SimulatedAnnealing(initial_temp = 30, final_temp = 0.001, alpha = 0.001, per_temp=1)


def cost_func(state, isCurrent):
    target_diff = numpy.array(state) - numpy.array(target_x)
    y_out, _ = compute_mu_and_sigma(x, state)
    diff =  y - y_out
    error = numpy.sum(numpy.absolute(diff))
    if (isCurrent):
        print("error",error, state)
    return error

sim_an.get_cost = cost_func
def neighbors_func(state, temp):
    #cov = numpy.random.rand(len(state), len(state)) 
    change = temp 
    cov = numpy.eye(len(state)) * change
    rtn = numpy.random.multivariate_normal(state, cov, 100)
    return rtn

sim_an.get_neighbors = neighbors_func

sim_an.anneal(initial_z)

