import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
# RENDER_MODE = None # selecione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000

env = gym.make("LunarLander-v3", render_mode=RENDER_MODE, 
               continuous=True, gravity=GRAVITY, 
               enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
               turbulence_power=TURBULENCE_POWER)

def check_successful_landing(observation):
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1
    on_landing_pad = abs(x) <= 0.2
    stable_velocity = vy > -0.1
    stable_orientation = abs(theta) < np.deg2rad(10)
    stable = stable_velocity and stable_orientation

    if legs_touching and on_landing_pad and stable:
        print("✅ Aterragem bem sucedida!")
        return True
    
    print("⚠️ Aterragem falhada!")
    return False

def simulate(steps=1000, seed=None, policy=None):
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ)
        observ, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    success = check_successful_landing(observ)
    return step, success

def get_perceptions(observation):
    return {
        "horizontal_position": observation[0],
        "vertical_position": observation[1],
        "horizontal_speed": observation[2],
        "vertical_speed": observation[3],
        "angle": observation[4],
        "angular_speed": observation[5],
        "left_leg_contact": observation[6],
        "right_leg_contact": observation[7],
    }

def apply_thrust(power=1.0):
    return np.array([power, 0], dtype=np.float64)

def rotate_left(power=0.5):
    return np.array([0, -power], dtype=np.float64)

def rotate_right(power=0.5):
    return np.array([0, power], dtype=np.float64)

def reactive_agent(observation):
    perceptions = get_perceptions(observation)
    action = np.array([0, 0], dtype=np.float64)
    while(1):
        if perceptions["vertical_speed"] < -0.2:
            action[0] = 1.0
        elif perceptions["vertical_speed"] < -0.1:
            action[0] = 0.3
        
        if perceptions["horizontal_position"] > 0.2:
            action += rotate_left(0.1)
        elif perceptions["horizontal_position"] < -0.2:
            action += rotate_right(0.1)
        
        if abs(perceptions["angle"]) > np.deg2rad(5):
            if perceptions["angle"] > 0:
                action += rotate_left(0.5)
            else:
                action += rotate_right(0.5)
        return np.clip(action, -1, 1)

def keyboard_agent(observation):
    action = np.array([0, 0], dtype=np.float64)
    keys = pygame.key.get_pressed()
    
    print('Observação:', observation)
    
    if keys[pygame.K_UP]:
        action[0] = 1.0
    if keys[pygame.K_LEFT]:
        action[1] = -0.5
    if keys[pygame.K_RIGHT]:
        action[1] = 0.5
    
    return np.clip(action, -1, 1)

success = 0.0
steps = 0.0
for i in range(EPISODES):
    st, su = simulate(steps=1000000, policy=reactive_agent)
    if su:
        steps += st
    success += su
    
    if success > 0:
        print('Média de passos das aterragens bem sucedidas:', steps / success)
    print('Taxa de sucesso:', (success / (i + 1)) * 100, '%')
