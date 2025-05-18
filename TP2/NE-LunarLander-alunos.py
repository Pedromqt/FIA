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

    
ELITE_SIZE = 1

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
    x, y, vx, vy, theta, vtheta, contact_left, contact_right = observation
    landed_successfully = check_successful_landing(observation)
    legs_touching = contact_left == 1 and contact_right == 1

    fitness = 0.0

    #Penalties and Rewards
    W_X_POS = 45.0
    W_VX = 65.0
    W_VY_DEVIATION = 85.0
    W_VY_POSITIVE_LOW = 160.0
    W_THETA = 80.0
    W_VTHETA = 55.0
    
    PROXIMITY_REWARD_SCALE = 200.0
    MAX_PROXIMITY_DISTANCE_EFFECTIVE = 2.8
    Y_PROXIMITY_WEIGHT = 1.75

    TARGET_IMPACT_VY = -0.05

    CONTROLLED_HOVER_BONUS_BASE = 300.0
    HOVER_ADDITIONAL_PRECISION_BONUS = 100.0
    HOVER_X_THRESHOLD = 0.07
    HOVER_Y_THRESHOLD = 0.12
    HOVER_VX_THRESHOLD = 0.07
    HOVER_VY_ERROR_THRESHOLD = 0.07
    HOVER_THETA_THRESHOLD_RAD = np.deg2rad(6.0)
    HOVER_VTHETA_THRESHOLD = 0.07

    # Crash Penalties
    CRASH_ADJ_ON_PAD = 200.0
    CRASH_ADJ_NEAR_PAD = 25.0
    CRASH_ADJ_FAR_PAD = -200.0
    CRASH_X_THRESHOLD_ON_PAD = 0.30
    CRASH_X_THRESHOLD_NEAR_PAD = 0.65

    # Crash condition penalty factors
    CRASH_X_PENALTY_FACTOR = 55.0
    CRASH_VX_PENALTY_FACTOR = 45.0
    CRASH_VY_PENALTY_FACTOR = 65.0
    CRASH_THETA_PENALTY_FACTOR = 45.0
    CRASH_VTHETA_PENALTY_FACTOR = 30.0

    # Successful Landing Bonuses
    SUCCESS_BONUS_BASE = 1300.0
    # Precision bonuses for clean landings
    SUCCESS_PRECISION_TOTAL_BONUS_CAP = 250.0
    SUCCESS_VX_CLEAN_FACTOR = 0.3
    SUCCESS_VY_CLEAN_FACTOR = 0.4
    SUCCESS_THETA_CLEAN_FACTOR = 0.3
    # Thresholds for "clean" landing
    VX_CLEAN_THRESHOLD = 0.03
    THETA_CLEAN_THRESHOLD_RAD = np.deg2rad(3.0)

    # Height Proximity Factor (0=high, 1=low, for scaling flight penalties)
    y_for_min_effect = 1.35 
    y_for_max_effect = 0.02
    if y > y_for_min_effect: height_proximity_scale = 0.0
    elif y < y_for_max_effect: height_proximity_scale = 1.0
    else: height_proximity_scale = (y_for_min_effect - y) / (y_for_min_effect - y_for_max_effect)

    #Penalties & Rewards During Flight
    current_flight_fitness = 0.0
    current_flight_fitness -= W_X_POS * (x**2) * (1 + 0.8 * height_proximity_scale)
    current_flight_fitness -= W_VX * (vx**2) * (1 + 1.3 * height_proximity_scale)
    
    ideal_vy_high = -0.50
    target_vy_flight = ideal_vy_high + (TARGET_IMPACT_VY - ideal_vy_high) * height_proximity_scale
    vy_flight_error = vy - target_vy_flight
    current_flight_fitness -= W_VY_DEVIATION * (vy_flight_error**2) * (1 + 1.3 * height_proximity_scale)
    if y < 0.5 and vy > 0.015:
        current_flight_fitness -= W_VY_POSITIVE_LOW * (vy**1.5) * (1 + 1.2 * height_proximity_scale)
        
    current_flight_fitness -= W_THETA * (theta**2) * (1 + 2.0 * height_proximity_scale)
    current_flight_fitness -= W_VTHETA * (vtheta**2) * (1 + 1.3 * height_proximity_scale)

    if not legs_touching:
        distance_to_pad_center = (x**2 + (y * Y_PROXIMITY_WEIGHT)**2)**0.5
        proximity_factor = max(0, 1 - distance_to_pad_center / MAX_PROXIMITY_DISTANCE_EFFECTIVE)
        current_flight_fitness += PROXIMITY_REWARD_SCALE * (proximity_factor**3)
    
    fitness = current_flight_fitness

    # Landing Points
    if legs_touching:
        if landed_successfully:
            fitness += SUCCESS_BONUS_BASE
            
            precision_bonus = 0
            precision_bonus += (SUCCESS_VX_CLEAN_FACTOR * SUCCESS_PRECISION_TOTAL_BONUS_CAP) * max(0, 1 - abs(vx) / VX_CLEAN_THRESHOLD)
            
            vy_success_error = vy - TARGET_IMPACT_VY
            denominator_vy_bonus = abs(TARGET_IMPACT_VY) if TARGET_IMPACT_VY != 0 else 0.01
            precision_bonus += (SUCCESS_VY_CLEAN_FACTOR * SUCCESS_PRECISION_TOTAL_BONUS_CAP) * max(0, 1 - abs(vy_success_error) / denominator_vy_bonus)
            
            precision_bonus += (SUCCESS_THETA_CLEAN_FACTOR * SUCCESS_PRECISION_TOTAL_BONUS_CAP) * max(0, 1 - abs(theta) / THETA_CLEAN_THRESHOLD_RAD)
            fitness += precision_bonus
        else:
            base_crash_score_adjustment = 0.0
            if abs(x) <= CRASH_X_THRESHOLD_ON_PAD: base_crash_score_adjustment = CRASH_ADJ_ON_PAD
            elif abs(x) <= CRASH_X_THRESHOLD_NEAR_PAD: base_crash_score_adjustment = CRASH_ADJ_NEAR_PAD
            else: base_crash_score_adjustment = CRASH_ADJ_FAR_PAD
            fitness += base_crash_score_adjustment
            
            # Penalties are subtracted
            fitness -= CRASH_X_PENALTY_FACTOR * (x**2) 
            fitness -= CRASH_VX_PENALTY_FACTOR * (vx**2)
            vy_impact_error = vy - TARGET_IMPACT_VY
            fitness -= CRASH_VY_PENALTY_FACTOR * (vy_impact_error**2)
            fitness -= CRASH_THETA_PENALTY_FACTOR * (theta**2)
            fitness -= CRASH_VTHETA_PENALTY_FACTOR * (vtheta**2)
    else:
        is_centered_x = abs(x) < HOVER_X_THRESHOLD
        is_low_enough_y = y < HOVER_Y_THRESHOLD 
        is_stable_vx = abs(vx) < HOVER_VX_THRESHOLD
        is_stable_vy = abs(vy - TARGET_IMPACT_VY) < HOVER_VY_ERROR_THRESHOLD
        is_stable_theta = abs(theta) < HOVER_THETA_THRESHOLD_RAD
        is_stable_vtheta = abs(vtheta) < HOVER_VTHETA_THRESHOLD

        if is_centered_x and is_low_enough_y and is_stable_vx and is_stable_vy and is_stable_theta and is_stable_vtheta:
            fitness += CONTROLLED_HOVER_BONUS_BASE
            hover_precision_score = (1 - abs(x)/HOVER_X_THRESHOLD) + \
                                    (1 - y/HOVER_Y_THRESHOLD) + \
                                    (1 - abs(vx)/HOVER_VX_THRESHOLD) + \
                                    (1 - abs(vy - TARGET_IMPACT_VY)/HOVER_VY_ERROR_THRESHOLD) + \
                                    (1 - abs(theta)/HOVER_THETA_THRESHOLD_RAD) + \
                                    (1 - abs(vtheta)/HOVER_VTHETA_THRESHOLD)
            fitness += (hover_precision_score / 6.0) * HOVER_ADDITIONAL_PRECISION_BONUS
    
    return fitness, landed_successfully

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
        ind['success'] = success

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
    #render_mode = 'human'
    render_mode = None
    if evolve:
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
        for i in range(30):    
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
