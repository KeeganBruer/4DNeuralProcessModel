import random
import math

class SimulatedAnnealing:
    def __init__(self, initial_temp = 90, final_temp = 0.1, alpha = 0.01):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha

    def anneal(self, initial_state):
        """Peforms simulated annealing to find a solution"""
        initial_temp = self.initial_temp
        final_temp = self.final_temp
        alpha = self.alpha
        
        current_temp = initial_temp

        # Start by initializing the current state with the initial state
        self.current_state = initial_state
        solution = self.current_state

        while current_temp > final_temp:
            neighbor = random.choice(self.get_neighbors(self.current_state))

            # Check if neighbor is best so far
            cost_diff = self.get_cost(self.current_state) - self.get_cost(neighbor)

            # if the new solution is better, accept it
            if cost_diff > 0:
                solution = neighbor
            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            else:
                if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                    solution = neighbor

            self.current_state = solution
            print(self.current_state)
            # decrement the temperature
            current_temp -= alpha

        return solution

    def get_cost(self, state):
        """Calculates cost of the argument state for your solution."""
        raise NotImplementedError
        
    def get_neighbors(self, state):
        """Returns neighbors of the argument state for your solution."""
        raise NotImplementedError
