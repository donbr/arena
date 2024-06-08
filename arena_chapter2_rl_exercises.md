## Chapter 2: Reinforcement Learning Exercises - A Hands-on Journey to Mastering RL

The `chapter2_rl/exercises` folder within the ARENA 3.0 repository is a practical exploration of the world of Reinforcement Learning (RL).  Through hands-on coding exercises, you'll progress from simple bandit problems to complex control tasks, gaining a deep understanding of core RL concepts, algorithms, and the intricacies of training agents to interact with environments and make optimal decisions.

**Part 1: Introduction to RL (`part1_intro_to_rl`)**

This section lays the groundwork for understanding RL, introducing the fundamental concepts, algorithms, and the OpenAI Gym framework for interacting with environments.

* **Files:** 
    * `solutions.py`: Contains solutions to the exercises.
    * `tests.py`: Provides test functions to verify your implementations.
    * `utils.py`: Offers helper functions for plotting reward curves and setting random seeds.

**Key Exercises:**

1. **The Multi-Armed Bandit:**
    * **OpenAI Gym Basics:** You'll become familiar with the OpenAI Gym library, a standardized interface for RL environments. You'll learn about:
        * The structure of a `gym.Env`, including methods like `step` (take an action and observe the result), `reset` (initialize the environment), and `render` (visualize the environment).
        * Discrete action and observation spaces, where the agent has a finite set of choices and receives discrete observations from the environment.
        * The `info` dictionary, used by environments to provide additional information not part of the standardized interface (e.g., debugging information).
    * **Implementing Agents:** You'll implement several agents to solve the multi-armed bandit problem:
        * `RandomAgent`:  Chooses arms randomly, establishing a baseline for performance.
        * `RewardAveraging`:  Keeps track of the average reward for each arm and chooses actions based on these estimates, introducing the concept of exploration vs. exploitation.
        * `CheatyMcCheater`:  A "cheating" agent that always chooses the best arm, providing a performance ceiling for comparison and validating the environment's correctness.
        * `UCBActionSelection`:  Implements the Upper Confidence Bound (UCB) action selection strategy, balancing exploration and exploitation using an uncertainty-based criterion.
    * **Measuring & Comparing Performance:** You'll learn how to measure and compare the performance of different agents, using metrics like average reward and the frequency of choosing the best arm. The section emphasizes the inherent variability in RL performance due to randomness and the importance of running multiple trials to get reliable estimates.
2. **Tabular RL & Policy Improvement:**
    * **Formalizing RL:**  This section introduces a more formal mathematical framework for RL, covering concepts like:
        * **Markov Processes:**  Environments where the future state depends only on the current state and action, not on past history.
        * **Value Function:**  A function that assigns a value to each state, representing the expected cumulative reward the agent can achieve from that state, following a specific policy.
        * **Bellman Equation:**  A recursive equation that connects the value of a state to the values of its successor states, capturing the core principle of dynamic programming in RL.
    * **Policy Evaluation & Improvement:** You'll learn about methods for evaluating the value of a given policy and iteratively improving it:
        * **Numerical Policy Evaluation:** Uses the Bellman equation to iteratively update the value function for a given policy until convergence.
        * **Policy Improvement:**  Generates a better policy by choosing actions greedily with respect to the current value function.
    * **Finding Optimal Policies:** You'll implement these algorithms and apply them to solve simple gridworld environments, showcasing how to find optimal policies for environments with known dynamics (i.e., the transition probabilities and reward function are available).

**Part 2: Q-Learning and DQN (`part2_q_learning_and_dqn`)**

This section builds upon the tabular RL concepts from the previous section, introducing **Q-learning**, a model-free algorithm that doesn't require knowledge of the environment's dynamics. You'll then extend this concept to **Deep Q-Learning (DQN)**, using a neural network to approximate the Q-function and solve the classic **CartPole** environment.

* **Files:** 
    * `play_cartpole.py`: Provides a simple script for manually controlling the CartPole environment using keyboard input.
    * `solutions.py`:  Contains solutions to the exercises.
    * `tests.py`:  Offers test functions to verify your implementations.
    * `utils.py`: Includes helper functions for managing random seeds, plotting rewards, and creating environments.

**Key Exercises:**

1. **Q-Learning:**
    * **Model-Free Learning:** Introduces Q-learning, a model-free algorithm that enables agents to learn optimal actions through direct interaction with the environment, without needing to know the transition probabilities or reward function.
    * **Q-Function:**  Explains the concept of the state-action value function (Q-function), which estimates the expected cumulative reward for taking a particular action in a given state and following the optimal policy thereafter.
    * **Bellman Optimality Equation & Update Rule:** Covers the Bellman optimality equation, which defines the optimal Q-function, and the Q-learning update rule for iteratively approximating it based on observed experiences.
    * **SARSA vs. Q-Learning:**  You'll implement the SARSA (State-Action-Reward-State-Action) algorithm, a variant of Q-learning that uses the agent's current policy to choose the next action for updating the Q-function. You'll compare SARSA and Q-learning on different environments, understanding their differences in exploration and exploitation.
2. **DQN:**
    * **Deep Q-Networks:** Extends Q-learning to handle continuous state spaces by using a deep neural network (the Deep Q-Network) to approximate the Q-function.
    * **Experience Replay:**  You'll implement a **replay buffer** to store past experiences and sample from them randomly during training, which helps break correlations in the data and improve learning stability.
    * **Target Network:** You'll implement a separate, slowly updated target network to calculate target values for the Q-learning updates, further stabilizing training.
    * **Epsilon-Greedy Policy:** You'll implement an epsilon-greedy policy based on the Q-network's predictions, balancing exploration and exploitation.
    * **DQN Agent:** You'll combine these components to create a DQN agent, train it on the CartPole environment, and visualize its performance using plots and animations.
    * **Debugging RL Agents:** The section emphasizes the importance of debugging in RL, providing tips and tricks for identifying and resolving issues:
        * **Probe Environments:**  You'll work with pre-defined probe environments with simplified dynamics to test specific aspects of your agent's behavior (e.g., learning from rewards, handling time delays, mapping observations to actions).
        * **Analyzing Transitions:** You'll inspect the transitions stored in the replay buffer to verify that the agent is correctly handling observations, actions, rewards, and termination states.
    * **Reward Shaping:** The section revisits the concept of reward shaping, modifying the CartPole environment's reward function to provide more frequent feedback to the agent and potentially speed up learning.

**Part 3: PPO (`part3_ppo`)**

This section introduces **Proximal Policy Optimization (PPO)**, a powerful and widely used policy gradient algorithm that offers advantages over DQN in terms of stability, sample efficiency, and performance in complex environments.

* **Files:** 
    * `atari_wrappers.py`: Contains various wrapper classes for Atari environments, handling pre-processing and specific game mechanics.
    * `play_breakout.py`: A script for manually playing the Breakout Atari game.
    * `play_mountaincar.py`: A script for manually playing the MountainCar environment.
    * `solutions.py` and `solutions_cts.py`: Contains solutions to the exercises, with the latter file dedicated to continuous action spaces.
    * `tests.py`: Provides test functions to verify your implementations.
    * `utils.py`: Includes helper functions for environment setup, random seed management, visualization, and reward calculation.

**Key Exercises:**

1. **Setting up the PPO Agent:**
    * **Actor-Critic Architecture:** Introduces the actor-critic architecture, where:
        * **The actor network** learns a policy that maps states to actions, aiming to maximize expected rewards.
        * **The critic network** learns a value function that estimates the expected cumulative reward from a given state, providing a baseline for evaluating the actor's policy.
    * **Generalized Advantage Estimation (GAE):**  You'll implement GAE, a method for efficiently estimating advantages, which represent how much better an action is than the average action taken in a given state. Advantages are used to guide the policy updates in PPO.
    * **Replay Memory:**  You'll create a `ReplayMemory` class to store experiences (state, action, reward, done, log probability of the action, value estimate) collected during the rollout phase.
    * **PPO Agent:** You'll build a `PPOAgent` class that encapsulates the actor and critic networks, the replay memory, and a `play_step` method for interacting with the environment and storing experiences.
2. **Learning Phase:**
    * **PPO Objective Function:**  You'll implement the PPO objective function, which consists of three key components:
        * **Clipped Surrogate Objective:**  Encourages the policy to improve while preventing drastic changes that could destabilize training.
        * **Value Function Loss:**  Trains the critic network to accurately estimate the value function.
        * **Entropy Bonus:**  Promotes exploration by encouraging the policy to have higher entropy (i.e., be less deterministic).
    * **Optimization & Learning Rate Scheduling:**  You'll implement a function to create an optimizer (Adam) and a learning rate scheduler that linearly decays the learning rate over the course of training.
3. **Training Loop:**
    * **Rollout & Learning Phases:** You'll build a training loop that alternates between:
        * **Rollout Phase:**  The agent interacts with the environment using its current policy, collecting experiences and storing them in the replay memory.
        * **Learning Phase:**  The agent samples minibatches of experiences from the replay memory and uses them to update its actor and critic networks, maximizing the PPO objective function.
    * **Training & Visualization:** You'll train your PPO agent on the CartPole environment, using Weights & Biases (WandB) to visualize training progress, log metrics like episode length and return, and optionally record videos of the agent's performance.
    * **Reward Shaping:**  You'll revisit the concept of reward shaping, modifying the CartPole environment's reward function to incentivize specific behaviors (e.g., spinning the pole) and observe how it affects the agent's learning process.
4. **Atari:**
    * **Extending PPO to Visual Domains:** This section adapts PPO to the more complex setting of Atari games, which have high-dimensional visual observations and require using a convolutional neural network (CNN).
    * **Shared Architectures:**  You'll implement a shared CNN architecture for the actor and critic networks, where the early layers extract features from the input images, and separate policy and value heads are used to map these features to actions and value estimates, respectively.
    * **Atari Wrappers:**  You'll learn about various wrapper classes commonly used for Atari environments, which handle preprocessing steps like frame skipping, image resizing, and reward clipping.
    * **Training on Breakout:**  You'll train your Atari PPO agent on the Breakout game, observing how it learns to control the paddle and break bricks to maximize its score.
5. **MuJoCo:**
    * **Continuous Action Spaces:**  This section extends PPO to handle continuous action spaces, where the agent can choose actions from a range of values rather than a discrete set.
    * **MuJoCo Physics Engine:** You'll be introduced to the MuJoCo physics engine, which provides realistic simulations of robotic and other physical systems.
    * **Normal Distribution for Actions:** You'll modify the actor network to output the parameters of a normal distribution (mean and log standard deviation) from which actions are sampled.
    * **Training on Hopper:**  You'll train your PPO agent on the Hopper environment, a challenging task that requires precise control of a simulated hopping robot.
* **Bonus:**
    * This section offers suggestions for extending your PPO implementation, including:
        * Implementing **Trust Region Policy Optimization (TRPO)**, a policy gradient algorithm that uses a constraint on the KL divergence between old and new policies to ensure stability.
        * Using a **long-term replay memory** to mitigate catastrophic forgetting.
        * **Vectorizing the advantage calculation** for increased efficiency.
        * Experimenting with other discrete environments (Acrobot, MountainCar, LunarLander).
        * Implementing reward shaping for continuous action spaces.
        * Exploring more complex environments like **Minigrid, Procgen, microRTS, and Megastep**.
        * Implementing **Multi-Agent PPO (MAPPO)** for training multiple agents simultaneously.

**Part 4: RLHF (`part4_rlhf`)**

This final section of the RL chapter brings together many of the concepts from earlier sections, demonstrating how RL can be applied to fine-tune transformer language models using human feedback. 

* **Files:**
    * `solutions.py`:  Contains solutions to the exercises.
    * `tests.py`:  Provides test functions to verify your implementations.

**Key Exercises:**

1. **RLHF on Transformer Language Models:**
    * **RL Framework for Transformers:** You'll learn how to apply the RL framework to autoregressive language models, where the agent is the model itself, actions are token predictions, and the environment is the process of generating text.
    * **Value Head:**  You'll implement a `TransformerWithValueHead` class, extending a standard transformer model with a **value head**. This head takes the last hidden state of the transformer as input and outputs a value estimate for each token, allowing the model to act as both an actor (generating text) and a critic (evaluating the generated text).
    * **Reward Functions:** You'll explore different reward functions for evaluating the model's generated text, including a simple character-counting reward function and a more complex sentiment-based reward function using a pre-trained sentiment classifier.
    * **RLHF Training Loop:**  You'll build a full RLHF training loop using the PPO algorithm, incorporating the clipped surrogate objective, value loss, entropy bonus, and a **KL penalty** term to prevent the model's policy from diverging too far from the reference model.
    * **Mode Collapse:**  You'll observe and understand the phenomenon of **mode collapse**, where the model gets stuck in a suboptimal solution that easily maximizes reward but lacks diversity or quality.
2. **Bonus:**
    * This section suggests various ways to extend and improve your RLHF implementation, including:
        * **Differential Learning Rates & Frozen Layers:** Experimenting with different learning rates for the base model and the value head, or freezing some layers of the base model during training, to prevent catastrophic forgetting and improve stability.
        * **Hyperparameter Sweeps & Adaptive KL Penalty:**  Performing hyperparameter sweeps to find optimal values for the KL penalty coefficient and other hyperparameters, and implementing an adaptive KL penalty that adjusts based on the observed KL divergence between the current and previous policies.
        * **TRL / trlX Library:**  Exploring dedicated libraries like TRL and trlX, which provide higher-level abstractions for implementing RLHF on transformers, simplifying the process and potentially offering better performance.
        * **Learning a Human Preference Reward Model:**  Training a separate reward model using supervised learning techniques, where human feedback (e.g., pairwise comparisons of text samples) is used to train a model that predicts human preferences.
        * **Mechanistic Interpretability of RLHF'd Models:**  Applying interpretability techniques to analyze how RLHF affects the model's internal representations and circuits, investigating whether it leads to the emergence of new behaviors or changes in existing circuits.

By the end of this chapter, you'll have a solid understanding of the core principles and algorithms of Reinforcement Learning, hands-on experience in implementing and training agents, and familiarity with the exciting field of RLHF for aligning language models with human preferences.
