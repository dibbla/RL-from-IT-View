from stable_baselines3 import PPO
import os
from env import singleEnv
import csv
# In cmd Admin
# tensorboard --logdir=logs

models_dir = "final_models"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
testing = False
csv_generation = True

env = singleEnv()
env.reset()
if testing or csv_generation:
    model_path = f"{models_dir}/380000.zip"
    model = PPO.load(model_path, env=env)
else:
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 20000
for i in range(60):
    if testing:
        obs = env.reset()
        while testing:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
    elif csv_generation:
        with open('expert_policy.csv', 'w', newline='') as file:
            obs = env.reset()
            writer = csv.writer(file)
            writer.writerow(["obs.no", "agent_x", "agent_y", "gem_x", "gem_y", "rewards", "action"])
            obs_no = 1
            done = True
            while done:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                writer.writerow([obs_no, obs[0], obs[1], obs[2], obs[3], reward, action])
                obs_no += 1
    else:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()