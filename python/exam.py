import numpy as np 

def iniat_population(population_size):
     return np.random.randint(8, size=(population_size, 8))

def calculate_fitness(population):
     fitness_vals =[]
     for x in population:
          penality =0
          for i in range (8):
               r=x[i]
               for j in range (8):
                    if i== j:
                         continue 
                    d=abs(i-j)
                    if x[j]in [r,r+d,r-d]:
                         penality += 1

          fitness_vals.append(penality)
     return -1 * np.array(fitness_vals)



def selection(population, fitness_vals):
    probs = fitness_vals.copy()
    probs += abs(probs.min()) + 1
    probs = probs/probs.sum()
    N = len(population)
    indices = np.arange(N)  
    selected_indecies = np.random.choice (indices, size = N,p= probs) 
    selected_population = population[selected_indecies]
    return selected_population


initial_population = iniat_population(4)
print("\ninitial population \n", initial_population)
fitness_vals = calculate_fitness(initial_population)

print("\nfitness\n", fitness_vals)
selected_population = selection(initial_population, fitness_vals)
print("\nSelected pop\n",selected_population)
            

     
    