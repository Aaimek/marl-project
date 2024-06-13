from gym.envs.registration import register
from envs.predator_prey import PredatorPreyEnv

register(
    id='PredatorPrey-v0',
    entry_point='envs.predator_prey:PredatorPreyEnv',
)
