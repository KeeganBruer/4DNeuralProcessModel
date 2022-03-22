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
            # decrement the temperature
            self.current_temp -= self.alpha
            ##get neighbor and cost difference
            neighbor = random.choice(self.get_neighbors(self.current_state))
            cost_diff = self.get_cost(self.current_state) - self.get_cost(neighbor)

            # if the new solution is better, accept it
            if cost_diff > 0:
                self.current_state = neighbor
                continue
            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            if random.uniform(0, 1) < math.exp(-cost_diff / self.current_temp):
                self.current_state = neighbor
                continue            
        return solution

    def get_cost(self, state):
        """Calculates cost of the current state."""
        raise NotImplementedError
        
    def get_neighbors(self, state):
        """Returns neighbors of the current state"""
        raise NotImplementedError
