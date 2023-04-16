from stable_baselines3 import PPO
import os
from env import singleEnv
# In cmd Admin
# tensorboard --logdir=logs

models_dir = "final_models"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
testing = True

env = singleEnv()
env.reset()
if testing:
    model_path = f"{models_dir}/150000.zip"
    model = PPO.load(model_path, env=env)
else:
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 50000
for i in range(60):
    if testing:
        obs = env.reset()
        while testing:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
    else:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()