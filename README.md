# RL-from-IT-View
A project repository discussing basic Reinforcement Learning algorithms from a Information Theory View

## Roadmap
- [x] Set up the maze environment
- [x] Generate expert trajectories with common baseline algorithms
- [x] Track MEC + Validation accuracy of expert trajectories using Brainome
- [ ] Run Behavior Clone on expert trajectories (no data-drifting)
- [ ] Run DQN on maze environment directly (data-drifting)
- [ ] More training with varying Memeory Equivalent Capacity (MEC)

## Environment setup
We use 
1. A customized grid world implemetation based on gym. 
2. `minigrid` based on `gymnasium`, which is the *"more modern"* version of OpenAI Gym.
