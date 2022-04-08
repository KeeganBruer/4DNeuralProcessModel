import random
import math

class SimulatedAnnealing:
    def __init__(self, initial_temp = 90, final_temp = 0.1, alpha = 0.1, per_temp=1):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.per_temp = per_temp

    def anneal(self, initial_state):
        """Peforms simulated annealing to find a solution"""        
        # Start by initializing the current temp and state
        self.current_temp = self.initial_temp
        self.current_state = initial_state

        while self.current_temp > self.final_temp:
            ##get neighbor and cost difference
            neighbors = self.get_neighbors(self.current_state, self.current_temp )
            curr_cost = self.get_cost(self.current_state, True) 
            for i in range(self.per_temp):
                neighbor = random.choice(neighbors)
                neighbor_cost = self.get_cost(neighbor, False)
                cost_diff = curr_cost - neighbor_cost
                try:
                    prob = math.exp(-(neighbor_cost-curr_cost) / self.current_temp)
                except OverflowError:
                    prob = 0 #If out of bounds don't accept solution.
                
                print("T ", cost_diff, self.current_temp, prob)
                # if the new solution is better, accept it
                if cost_diff >= 0:
                    self.current_state = neighbor    
                    curr_cost = neighbor_cost          
                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
                elif random.uniform(0, 1) < prob:
                    self.current_state = neighbor
                    curr_cost = neighbor_cost #Save neighbor cost instead of recalculating cost of new current state 
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
