import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = True
WIND_POWER = 10.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
#RENDER_MODE = 'human'
RENDER_MODE = None #seleccione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000

env = gym.make("LunarLander-v3", render_mode =RENDER_MODE, 
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

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        print("✅ Aterragem bem sucedida!")
        return True

    print("⚠️ Aterragem falhada!")        
    return False
        
def simulate(steps=1000,seed=None, policy = None):    
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ)

        observ, _, term, trunc, _ = env.step(action)

        if term or trunc:
            break

    success = check_successful_landing(observ)
    return step, success



#Perceptions
##TODO: Defina as suas perceções aqui
def perceptions(observation):
    
    x = observation[0]
    y = observation[1]
    vx = observation[2]
    vy = observation[3]
    theta = observation[4]
    vtheta = observation[5]

    return x, y, vx, vy, theta, vtheta


#Actions
def calc_action(x, y, vx, vy, theta, vtheta):
    h_motors = 0 
    if(ENABLE_WIND):
        anguloLim = 0.28
    else: 
        anguloLim = 0.2
        
    # Motores Horizontais de acordo com a velocidade da nave:
    if vx > 0.175:
        h_motors = -1
    elif vx < -0.175:
        h_motors = 1
        
    # Motor Vertical de acordo com a velocidade da nave:
    if vy < -0.08 and y > 0.055:
        v_motor = 1
    else: 
        v_motor = 0
    
    # Motores horizontais de acordo com a posição da nave:
    if y >= 0.91:  
        if x > 0.0305 and vx > -0.04:  
            h_motors = -0.85
        elif x < -0.0305 and vx < 0.04:  
            h_motors = 0.85
    else:
        if x > 0.0305 and vx > -0.04:  
            h_motors = -0.95
        elif x < -0.0305 and vx < 0.04:  
            h_motors = 0.95
            
    # Motores Horizontais de acordo com o angulo e velocidade de rotação da nave:
    if theta < -anguloLim and y > 0.1:
        h_motors = -0.9
    elif theta > anguloLim  and y > 0.1:
        h_motors = 0.9

    if vtheta < -anguloLim and y > 0.1:
        h_motors = -1
    elif vtheta > anguloLim and y > 0.1:
        h_motors = 1 


    return v_motor, h_motors



def reactive_agent(observation):
   
    ##TODO: Implemente aqui o seu agente reativo
    ##Substitua a linha abaixo pela sua implementação
    
    x,y,vx,vy,theta,vtheta = perceptions(observation)  
    v_motor,h_motors = calc_action(x,y,vx,vy,theta,vtheta)

    #action = env.action_space.sample()
    
    return [v_motor, h_motors]
    
    
def keyboard_agent(observation):
    action = [0,0] 
    keys = pygame.key.get_pressed()
    
    print('observação:',observation)

    if keys[pygame.K_UP]:  
        action =+ np.array([1,0])
    if keys[pygame.K_LEFT]:  
        action =+ np.array( [0,-1])
    if keys[pygame.K_RIGHT]: 
        action =+ np.array([0,1])

    return action
    

success = 0.0
steps = 0.0
for i in range(EPISODES):
    st, su = simulate(steps=1000000, policy=reactive_agent)

    if su:
        steps += st
    success += su
    
    if su>0:
        print('Média de passos das aterragens bem sucedidas:', steps/success*100)
    print('Taxa de sucesso:', success/(i+1)*100)
    
