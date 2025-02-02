import gymnasium as gym
from gym import envs

#all_envs = envs.registry.all()
#env_ids = [env_spec.id for env_spec in all_envs]
#pprint(sorted(env_ids))
for key in envs.registry.keys():
    print(key)


