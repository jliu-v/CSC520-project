# CSC520 Project  
Pacman Game using Minimax with Alpha Beta Pruning, Semi-online Search, and Reinforcement Learning
## Prerequisites
- Download this project
- Install OpenAI Baselines library
  - Download and unzip the _baseline_ repository from this link: https://github.com/openai/baselines/tree/tf2
  - Change directory into the `baseline` directory and run `pip install -e .`
      - If you get errors, ensure prerequisites of the OpenAI Baseline project are met and TensorFlow is installed: https://github.com/openai/baselines/tree/tf2#prerequisites

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