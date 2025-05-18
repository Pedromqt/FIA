import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
EPISODES = 1000
STEPS = 500

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100
PROB_CROSSOVER = 0.9

  
PROB_MUTATION = 1.0/GENOTYPE_SIZE
STD_DEV = 0.1


ELITE_SIZE = 0

def network(shape, observation,ind):
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x


def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        return True
    return False

def objective_function(observation):
    # Extract state variables
    x = observation[0]         # Horizontal position (0 is center of landing pad)
    y = observation[1]         # Vertical position (higher is better for altitude)
    vx = observation[2]        # Horizontal velocity
    vy = observation[3]        # Vertical velocity
    theta = observation[4]     # Angle
    vtheta = observation[5]    # Angular velocity
    contact_left = observation[6]  # Left leg contact
    contact_right = observation[7] # Right leg contact
    
    # Check landing status
    landed = check_successful_landing(observation)
    legs_touching = contact_left == 1 and contact_right == 1
    
    # Base fitness (will be negative unless landing is successful)
    fitness = 0
    
    # === POSITION REWARDS/PENALTIES ===
    # Horizontal centering - more penalty as we get farther from center
    horizontal_penalty = abs(x) * (1 + abs(x))  # Quadratic penalty for being off-center
    
    # Height penalty - increases as lander gets closer to ground to encourage precision
    height_factor = max(0.2, min(1.0, 1.0 - y/1.0)) if y > 0 else 1.0
    
    # === VELOCITY REWARDS/PENALTIES ===
    # Horizontal velocity - should approach zero as we near the ground
    vx_penalty = abs(vx) * (2.0 + height_factor * 8.0)
    
    # Vertical velocity - should be slow and controlled, especially near ground
    # Allow faster descent when high, require slower descent when close to ground
    ideal_vy = min(-0.05, -0.2 * y) if y > 0.2 else -0.05
    vy_penalty = abs(vy - ideal_vy) * (3.0 + height_factor * 12.0)
    
    # === ORIENTATION REWARDS/PENALTIES ===
    # Angle penalty - lander should be upright
    angle_penalty = abs(theta) * (20.0 + height_factor * 30.0)
    
    # Angular velocity penalty - should not be spinning
    angular_velocity_penalty = abs(vtheta) * 10.0
    
    # === CALCULATE FITNESS ===
    # Apply penalties
    fitness -= (
        40 * horizontal_penalty +
        70 * vx_penalty +
        80 * vy_penalty +
        80 * angle_penalty +
        50 * angular_velocity_penalty
    )
    
    # Distance-based reward - encourage getting closer to landing pad
    if not legs_touching:
        # Calculate distance to landing pad center
        distance_to_pad = ((x ** 2) + (y ** 2)) ** 0.5
        # Reward for being closer to the landing pad (max 50)
        proximity_reward = 50 * max(0, 1 - (distance_to_pad / 3.0))
        fitness += proximity_reward
    
    # Add bonuses for good flight characteristics
    if abs(vx) < 0.1:
        fitness += 10  # Bonus for minimal horizontal velocity
        
    if abs(theta) < 0.05:
        fitness += 15  # Bonus for being nearly upright
    
    if -0.2 < vy < 0:
        fitness += 20  # Bonus for appropriate descent rate
    
    # Touching ground penalties/rewards
    if legs_touching and not landed:
        # Crashed but touched with legs
        fitness -= 50  # Penalty for crashing
        
        # But still give some credit for almost landing
        if abs(x) < 0.4:  # Close to the pad
            fitness += 20
            
        if abs(theta) < 0.2:  # Nearly upright
            fitness += 15

    landing_bonus=0
    # Major bonus for successful landing
    if landed:
        # Base landing bonus
        landing_bonus = 1000
        
    # Additional bonus for clean landing (low velocities)
    landing_bonus += 200 * (1 - min(1.0, (abs(vx) + abs(vy)) / 0.5))
    
    # Additional bonus for good orientation
    landing_bonus += 100 * (1 - min(1.0, abs(theta) / 0.2))
        
    fitness += landing_bonus
    
    return fitness, landed



def simulate(genotype, render_mode = None, seed=None, env = None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode=render_mode,
                       continuous=True, gravity=GRAVITY,
                       enable_wind=ENABLE_WIND, wind_power=WIND_POWER,
                       turbulence_power=TURBULENCE_POWER)

    observation, info = env.reset(seed=seed)

    for _ in range(STEPS):
        prev_observation = observation
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        

        if terminated == True or truncated == True:
            break

    if env_was_none:
        env.close()

    return objective_function(prev_observation)

def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes
    
    env = gym.make("LunarLander-v3", render_mode =None, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
    while True:
        ind = evaluationQueue.get()
        if ind is None:
            break
        fit, success = simulate(ind['genotype'], seed=None, env=env)
        ind['fitness'] = fit
        ind['success'] = success  # ESSENCIAL!

        evaluatedQueue.put(ind)
    env.close()


def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop


def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def parent_selection(population,k=3):
    # Tournament selection
    selected = random.sample(population, k)
    selected.sort(key=lambda x: x['fitness'], reverse=True)
    return copy.deepcopy(selected[0])


def crossover(p1, p2):
    child = {'genotype' : [], 'fitness': None}
    for gene1, gene2 in zip(p1['genotype'], p2['genotype']):
        if random.random() < 0.5:
            child['genotype'].append(gene1)
        else:
            child['genotype'].append(gene2)
    return child


def mutation(p):
    for i in range(len(p['genotype'])):
        if random.random() < PROB_MUTATION:
            p['genotype'][i] += random.gauss(0, STD_DEV)
    return p


def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution():
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    #Create initial population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)
            else:
                ni = parent_selection(population)
            ni = mutation(ni)
            offspring.append(ni)
            
        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        
        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        success_rate = sum([ind.get('success', 0) for ind in population]) / POPULATION_SIZE
        print(f'Best of generation {gen}: {best[1]} | Success Rate: {success_rate:.2f}')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return bests


def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests


if __name__ == '__main__':
    evolve = True
    print(f"{PROB_MUTATION}- Prob Mutation")
    #render_mode = 'human'
    render_mode = None
    if evolve:
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
        for i in range(5):    
            random.seed(seeds[i])
            bests = evolution()
            with open(f'log{i}.txt', 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')
    else:
        #validate individual
        bests = load_bests('log4.txt')
        b = bests[-1]
        SHAPE = b[1]
        ind = b[2]
        ind = {'genotype': ind, 'fitness': None}
        ntests = 1000
        fit, success = 0, 0
        for i in range(1,ntests+1):
            print(f"{i}")
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
            fit += f
            success += s
        
        print(fit/ntests, success/ntests)
