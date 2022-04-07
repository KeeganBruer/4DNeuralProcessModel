import random
import math

class SimulatedAnnealing:
    def __init__(self, initial_temp = 90, final_temp = 0.1, alpha = 0.01):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha

    def anneal(self, initial_state):
        """Peforms simulated annealing to find a solution"""        
        # Start by initializing the current temp and state
        self.current_temp = self.initial_temp
        self.current_state = initial_state

        while self.current_temp > self.final_temp:
            ##get neighbor and cost difference
            neighbor = random.choice(self.get_neighbors(self.current_state, self.current_temp ))
            cost_diff = self.get_cost(self.current_state, True) - self.get_cost(neighbor, False)

            # if the new solution is better, accept it
            if cost_diff > 0:
                self.current_state = neighbor                
            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            elif random.uniform(0, 1) < math.exp(cost_diff / self.current_temp):
                self.current_state = neighbor
            # decrement the temperature
            self.current_temp -= self.alpha
        return self.current_state

    def get_cost(self, state, isCurrent):
        """Calculates cost of the current state."""
        raise NotImplementedError
        
    def get_neighbors(self, state, temp):
        """Returns neighbors of the current state"""
        raise NotImplementedError
if (__name__ == "__main__"):
    sim_an = SimulatedAnnealing()
    def cost(state):
        print(abs(state["cost"]))
        return abs(state["cost"])
    def neighbor(state, temp):
        neighbors = []
        neighbors.append({
            "cost":state["cost"]-1
        })
        neighbors.append({
            "cost":state["cost"]+1
        })
        

        return neighbors
    sim_an.get_cost = cost
    sim_an.get_neighbors = neighbor
    print("solution {}".format(sim_an.anneal({"cost":10})))
