# CSC520 Project  
Pacman Game using Minimax with Alpha Beta Pruning, Semi-online Search, and Reinforcement Learning
## Team members
- Tanya Chu (wchu2)
- Junfeng Liu (jliu85)
- Fangtong Zhou (fzhou)
## Source
This Pacman game environment was developed by John DeNero, Dan Klein, Pieter Abbeel, and many others for the [UC Berkeley](http://ai.berkeley.edu) CS188 course. See https://inst.eecs.berkeley.edu/~cs188/fa18/project2.html.
## Prerequisites
- Download this project
- Install OpenAI Baselines library
  - Download and unzip the _baselines_ repository from this link: https://github.com/openai/baselines/tree/tf2
  - Change directory into the `baselines` directory and run `pip install -e .`
      - If you get errors, ensure prerequisites of the OpenAI Baselines project are met and TensorFlow is installed: https://github.com/openai/baselines/tree/tf2#prerequisites

## Run Game
- To run the game in interactive mode, run `python pacman.py`
  - In this mode, users can play a classic game of Pacman using keyboard keys)
- In our experiments, we ran the game with agents using this command: `python pacman.py -p AlphaBetaAgent -l smallClassic -k 1 -g DirectionalGhost -n 350 -q`
  - `-p` is the pacman agent. We implemented and used the `AlphaBetaAgent`.
  - `-l` is the layout of the game. We chose to use the simple `smallClassic` layout.
  - `-k` is the number of ghosts in the game. We ran it with 1 ghost as the opponent.
  - `-g` is the ghost agent. We chose to use the `DirectionalGhost` agent which decide on their actions based on their Manhattan distance to Pacman.
  - `-n` is the number of game runs. We ran 350 games for the experiment.
  - `-q` is quietTextGraphics mode which generates minimal output and no graphics. This allows us to run the experiments faster and with less overhead. Users can leave this out and run a smaller number of games to see the agent in action.

## Contributions
As mentioned earlier, this project game environment was sourced. Here are our modifications and contributions to the environment for our project:
- MultiAgents.py
  - Function scoreEvaluationFunction()
    - Implementation baseline agent accessing scores of legal actions
  - Class AlphaBetaAgent()
    - Implementation of Alpha-Beta Pruning with depth limited semi-online search
    - Implementation of search path construction within the depth limit
    - Implementation of RL models (DQN/PPO) initialization and training
- contrib/util.py
    - Implementation of state perception function
- game.py
  - Game.run()
    - Implementation of adding agent experience to replay buffer
    - Implementation of triggering RL training 
    - Implementation of recording game results
- pacman.py
  - ClassicGameRules.newGame()
    - Implementation of RL model instantiation and hyper-parameters
