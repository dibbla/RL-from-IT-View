# Analysing RL from IT View
A project repository discussing basic Reinforcement Learning algorithms from a Information Theory View

## Roadmap
- [x] Set up the maze environment
- [x] Generate expert trajectories with common baseline algorithms
- [x] Track MEC + Validation accuracy of expert trajectories using Brainome
- [x] Run Behavior Clone on expert trajectories (no data-drifting)
- [x] Run DQN on maze environment directly (data-drifting)
- [x] More training with varying Memeory Equivalent Capacity (MEC)

## Environment setup
We use 
1. A customized grid world implemetation based on gym. 
2. `minigrid` based on `gymnasium`, which is the *"more modern"* version of OpenAI Gym.

To run this code, we recommand:
- torch==1.13.0
- gym==0.21
- gymnasium
- minigrid

## Supplemental Material: Experimental Design of BC and DQN Analysis
Behavioral clones offer an effective supervised reinforcement learning technique assuming access to large datasets and small action space. However, even with access to a corpus of expert data, behavioral clones fail to generalize solutions when given adversarial examples or data drift. Stochastic environments often result in cases that are not seen in training data. As traditional behavioral clones are meant to imitate only, they fail to handle edge cases. Competitive agents in a multi-agent environment magnify this problem as unexpected actions occur with no foreseeable counteraction by the clone. Our team addresses model decay due to data drift by generalizing behavioral clones through reducing equivalent memory capacity allowing the models to handle data drift and adversarial examples effectively. 

Imitation learning (IL) has been widely adopted in autonomous robtics as a way to bridge the gap between designed reward functions in traditional RL and the intended trajectory of an expert demonstration - data is often easier to acquire than accurate reward functions. Since large datasets from real world demonstrations of robotic behavior is hard to collect and leads to mechanical failures, engineers have turned to behavioral cloning as an IL method that does not require environmental interactions.

Behavioral Cloning (BC) has two assumptions: the expert demonstration provided is optimal, and the demonstrations encompass the entire state-action space. Assumption one is obtained through recording expert human behavior, training an agent to convergence, or creating a computational solver. Assumption two is either obtained through a perfect solver mapping every state-action space, or generating a large enough corpus of expert agent data that the dataset is assumed to contain every action-state space by probability (verified through validation set). 

Regardless of whether behavioral cloning memorizes the expert behavior, the generalizability of the behavior being cloned may not be accurate. Since the traditional goal is to mimic human behavior, human intentions are often overlooked. A naive solution is to cope via more data, which has been rather effective in low dimensional tasks (Atari Games). While this may solve the intended human objective, adding any stochastic element into the environment generally causes it to drastically reduce in accuracy. Evidentally, in order for behavioral cloning to progress as an effective imitaton learning algorithm, we must first solve generalizability.

Neural networks trained on the feature classification to 100% accuracy with a network bit size equivalent to the memory equivalent capacity (MEC) of the dataset has effectively memorized the data. Rather than working as a universal function aproximator, the neural network overfits, becoming a verbose hashtable where its generalization ratio is 1:1 bits per bit. 
We propose that forcing generalization through MEC reduction rather than large dataset training, will allow supervised models to handle data drift more effectively than traditional models. 
To rigourously test our BC, we provide a simple gem seeking environment where the agent has an initial random location in a 10 by 10 grid and moves toward a fixed position gem in the center of the grid. The agent has a discrete action space of the cardinal directions and stopping. On this controlled environment we test two agents, our experimental behavioral clone and a controlled expert policy.

We then create a synthetic corpus of every state space using both a computational solver and a expert PPO agent trained on the environemnt (non deterministic behavior due to discrete action space prevents effective cloning on simple gem environment from PPO, issue not seen in Atari Breakout cloning). Using Brainome we found the MEC of the CSV file storing the observation-action space of the expert policy with uniform distribution of actions. Starting at the memorized model with an MEC of 77, we iteratively trained network architectures of decreasing size until convergence. To test whether the networks generalized the task of finding the gem at the center of the grid to gems with randomized positions, we ran the trained models on drift I (within 1 block of center) and drift II (within 2 blocks of center) environments. Maximizing the accuracy-mec tradeoff optimzies for task generalization.

Above the BC validation accuracy has a positive nonlinear relationship with MEC where a 63 bit MEC has a 1.2:1 bits/bit, 100% validation accuracy, and 93% drift accuracy, surpassing the PPO and solver expert policy. Since this point on the curve maximizes the accuracy-MEC tradeoff, it has the highest task generalization shown empiracally through our drift metrics.
Consequently, the behavioral clone has higher resilience to data drift when the network is trained for generalization rather than memorization.  
 
*Memory Equivalent Capacity* - A machine learner’s intellectual capacity is memory-equivalent to N bits when the machine learner is able to represent all 2^N binary labeling functions of N uniformly random inputs. MEC is measurable via the shannon entropy of the target data - a measurement of the minimum complexity of the modeling space given an alphabet and a probability distribution over the alphabet. 

*Data Drift* - a variation in the production data from the data that was used to test and validate the model before deploying it in production.
