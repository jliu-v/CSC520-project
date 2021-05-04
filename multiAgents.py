# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np
import tensorflow as tf
from operator import attrgetter
from contrib.util import MyReplayBuffer


from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState, agentIndex):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    # directionsList = ['West', 'East', 'North', 'South', 'Stop']
    directionsList = ['North', 'East', 'South', 'West', 'Stop']
    legalMoves = currentGameState.getLegalActions(agentIndex)
    # print("Legal Moves:", legalMoves)

    scores = []
    for action in directionsList:
        if action in legalMoves:
            successorGameState = currentGameState.generateSuccessor(agentIndex, action)
            scores.append(successorGameState.getScore())
        else:
            scores.append(np.nan)

    return scores
    # return currentGameState.getScore()


class StateNode:
    def __init__(self, action, score, gameState):
        self.action = action
        self.score = score
        self.gameState = gameState
        self.successors = []


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        # self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.path = []
        self.previousLocs = [None, None]

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning
    """

    def __init__(self, **kwargs):
        super(AlphaBetaAgent, self).__init__(**kwargs)

        self.model_type = None
        self.rl_model = None
        self.replay_buffer = None

        from game import Actions
        self.action_mapping = {v[0]: i for i, v in enumerate(Actions._directionsAsList)}
        self.action_mapping_reverse = {v: k for k, v in self.action_mapping.items()}

    def evaluation_fn(self, currentGameState):
        if self.model_type == "dqn":
            from contrib.util import state_to_obs_tensor
            obs = state_to_obs_tensor(currentGameState)
            scores = self.rl_model.q_network(np.expand_dims(obs, axis=0))[0].numpy()

        elif self.model_type == "ppo":
            from contrib.util import state_to_obs_tensor
            obs = state_to_obs_tensor(currentGameState)
            latent = self.rl_model.train_model.policy_network(np.expand_dims(obs, axis=0))
            pd, pi = self.rl_model.train_model.pdtype.pdfromlatent(latent)
            scores = pi[0].numpy()
        else:
            scores = scoreEvaluationFunction(currentGameState, 0)

        legalMoves = currentGameState.getLegalActions()

        for action in self.action_mapping.keys():
            if action not in legalMoves:
                scores[self.action_mapping[action]] = np.nan

        return scores

    # def create_rl_model(self, model_type="dqn", observation_shape, network='cnn', lr=1e-3, grad_norm_clipping=10, gamma=1.0, param_noise=False):
    def create_rl_model(self, model_type="dqn", model_args={}):
        self.model_type = model_type
        num_actions=len(self.action_mapping)

        if model_type == "dqn":
            from baselines.deepq.deepq_learner import DEEPQ
            from baselines.deepq.models import build_q_func
            q_func = build_q_func(model_args.pop("network", "cnn"), hiddens=[16])

            self.rl_model = DEEPQ(q_func=q_func, num_actions=num_actions, **model_args)

            self.update_target_network()
        elif model_type == "ppo":
            from baselines.ppo2.model import Model as PPO
            from baselines.common.models import get_network_builder
            import gym
            ac_space = gym.spaces.Discrete(num_actions)
            network = model_args.pop("network", "cnn")
            observation_shape = model_args.pop("observation_shape", "cnn")

            if isinstance(network, str):
                network_type = network
                policy_network_fn = get_network_builder(network_type)()
                network = policy_network_fn(observation_shape)
            self.lr = model_args.pop('lr', 1e-3)
            self.rl_model = PPO(ac_space=ac_space, policy_network=network, **model_args)

    def create_replay_buffer(self, buffer_size):
        self.replay_buffer = MyReplayBuffer(buffer_size)

    def train_rl_model(self, batch_size, seq_length=None):
        if self.model_type == "dqn":
            # when using dqn, we only need to sample atomic states
            obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(batch_size)
            weights, batch_idxes = np.ones_like(rewards), None
            obses_t, obses_tp1 = tf.constant(obses_t), tf.constant(obses_tp1)
            actions, rewards, dones = tf.constant(actions), tf.constant(rewards), tf.constant(dones)
            weights = tf.constant(weights)
            td_errors = self.rl_model.train(obses_t, actions, rewards, obses_tp1, dones, weights)
            return td_errors
        elif self.model_type == "ppo":
            # when using ppo, we need to sample sequences
            seq_length = seq_length or 200
            start = 0 if self.replay_buffer._next_idx >= len(self.replay_buffer._storage) else -self.replay_buffer._max_size + self.replay_buffer._next_idx
            end = self.replay_buffer._next_idx - seq_length + 1
            sample_index = np.random.choice(np.arange(start, end), size=batch_size)

            # Note: variation of runner.run() to collect and process experiences
            # Here, we init the lists that will contain the mb of experiences
            # np_replay_buffer = np.array(self.replay_buffer._storage)
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
            last_obs = None
            for i in range(seq_length):
                # obs, act, reward, value, neglogp, done, obs1 = np_replay_buffer[sample_index+1].T
                obs, act, reward, value, neglogp, done, obs1 = [],[],[],[],[],[],[]
                for j in sample_index:
                    xobs, xact, xreward, xvalue, xneglogp, xdone, xobs1 = self.replay_buffer._storage[i+j]
                    obs.append(xobs)
                    act.append(xact)
                    reward.append(xreward)
                    value.append(xvalue)
                    neglogp.append(xneglogp)
                    done.append(xdone)

                    if i == seq_length - 1:
                        obs1.append(obs1)

                mb_obs.append(obs)
                mb_actions.append(act)
                mb_rewards.append(reward)
                mb_values.append(value)
                mb_neglogpacs.append(neglogp)
                mb_dones.append(done)

                if i == seq_length - 1:
                    last_obs = obs1

            # batch of steps to batch of rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            last_values = self.rl_model.value(tf.constant(last_obs))._numpy()

            print(last_values)

    def update_target_network(self):
        self.rl_model.update_target()

    def generateMaxNode(self, alpha, beta, state, depth):
        """
        Generate a max node in the tree
        """
        # # if current state is win or lose, return game state score
        # if state.isWin() or state.isLose():
        #     return state.getScore()
        #
        # # if reached depth limit, get min predicted score
        # if depth == 0:
        #     predicted_scores = scoreEvaluationFunction(state)
        #     return np.nanmin(predicted_scores)
        #
        # max_score = float('-inf')
        #
        # # get legal actions of current state
        # for action in state.getLegalActions():
        #     new_score = self.generateMinNode(alpha, beta, state.generateSuccessor(self.index, action), depth - 1)
        #     max_score = max(max_score, new_score)
        #
        #     # prune
        #     if max_score >= beta:
        #         return max_score
        #
        #     # set alpha if not pruned
        #     alpha = max(max_score, alpha)
        #
        # return max_score

        # if current state is win or lose, return game state score
        if state.gameState.isWin() or state.gameState.isLose():
            state.score = state.gameState.getScore()
            return state

        # if reached depth limit, get max predicted score
        if depth == 0:
            # predicted_scores = scoreEvaluationFunction(state.gameState, 0)
            predicted_scores = self.evaluation_fn(state.gameState)
            state.score = np.nanmax(predicted_scores)
            # print("depth limit action: ", state.action, " score: ", state.score)
            return state

        # get legal pacman actions of current state
        for action in state.gameState.getLegalActions(0):
            new_state = StateNode(action, float('inf'), state.gameState.generateSuccessor(0, action))
            new_state = self.generateMinNode(alpha, beta, new_state, depth-1)
            # print("successor")
            # print("action: ", action, " score: ", new_state.score)
            state.successors.append(new_state)
            if new_state.score > state.score:
                state.score = new_state.score

            # prune
            if state.score >= beta:
                # print("pruned ", state.score, " >= ", beta)
                return state

            # set alpha if not pruned
            alpha = max(state.score, alpha)
        # for s in state.successors:
        #     print("Successor action: ", s.action, " score: ", s.score)
        return state

    def generateMinNode(self, alpha, beta, state, depth):
        """
        Generate a min node in the tree
        """
        # # if current state is win or lose, return game state score
        # if state.isWin() or state.isLose():
        #     return state.getScore()
        #
        # # if reached depth limit, get max predicted score
        # if depth == 0:
        #     predicted_scores = scoreEvaluationFunction(state)
        #     return np.nanmax(predicted_scores)
        #
        # min_score = float("inf")
        #
        # # get legal actions of current state
        # for action in state.getLegalActions():
        #     new_score = self.generateMaxNode(alpha, beta, state.generateSuccessor(self.index, action), depth - 1)
        #     min_score = min(min_score, new_score)
        #
        #     # prune
        #     if min_score <= alpha:
        #         return min_score
        #
        #     # set beta if not pruned
        #     beta = min(min_score, beta)
        #
        # return min_score

        # if current state is win or lose, return game state score
        if state.gameState.isWin() or state.gameState.isLose():
            state.score = state.gameState.getScore()
            return state

        # if reached depth limit, get min predicted score
        if depth == 0:
            #predicted_scores = scoreEvaluationFunction(state.gameState, 1)
            predicted_scores = self.evaluation_fn(state.gameState)
            state.score = np.nanmin(predicted_scores)
            # print("depth limit score ", state.score)
            return state

        # get legal ghost actions of current state
        for action in state.gameState.getLegalActions(1):
            new_state = StateNode(action, float('-inf'), state.gameState.generateSuccessor(1, action))
            new_state = self.generateMaxNode(alpha, beta, new_state, depth - 1)
            # print("successor")
            # print("action: ", action, " score: ", new_state.score)
            state.successors.append(new_state)
            if new_state.score < state.score:
                state.score = new_state.score

            # prune
            if state.score <= alpha:
                # print("pruned ", state.score, " <= ", alpha)
                return state

            # set beta if not pruned
            beta = min(state.score, beta)
        # for s in state.successors:
        #     print("Successor action: ", s.action, " score: ", s.score)
        return state

    def generatePath(self, node):
        """
        Generate a path based on the generated minimax tree
        """
        is_max = True

        while node.successors:
            # get scores of successors
            scores = [s.score for s in node.successors]
            if is_max:
                # get maximum score and all nodes with the max score
                max_score = max(scores)
                best_indices = [index for index in range(len(scores)) if scores[index] == max_score]
                # get the best state out of all max states
                chosen_index = best_indices.pop(0)
                max_state = node.successors[chosen_index]
                # check if chosen state puts pacman in the previous position
                pos = max_state.gameState.getPacmanPosition()
                prev = self.previousLocs.pop(0)
                if len(node.successors) > 1 and len(best_indices) > 0 and\
                        (max_state.gameState.isLose() or pos == prev):
                    # get the next max state
                    # print("repeated location", pos)
                    chosen_index = best_indices.pop(0)
                    max_state = node.successors[chosen_index]
                self.previousLocs.append(max_state.gameState.getPacmanPosition())
                self.path.append(max_state.action)
                node = max_state

                # max_state = max(node.successors, key=lambda s: s.score)
                # pos = max_state.gameState.getPacmanPosition()
                # prev = self.previousLocs.pop(0)
                # if len(node.successors) > 1 and (max_state.gameState.isLose() or pos == prev):
                #     print("repeated location", pos)
                #     node.successors.remove(max_state)
                #     max_state = max(node.successors, key=lambda s: s.score)
                # self.previousLocs.append(max_state.gameState.getPacmanPosition())
                # self.path.append(max_state.action)
                # node = max_state

                # max_score = max(scores)
                # best_indices = [index for index in range(len(scores)) if scores[index] == max_score]
                # chosen_index = random.choice(best_indices)  # Pick randomly among the best
                # node = node.successors[chosen_index]
                # self.path.append(node.action)
            else:
                # print("ghost:", scores)
                # bestProb = 0.8
                # min_score = min(scores)
                # best_states = [node for node in node.successors if node.score == min_score]
                # dist = util.Counter()
                # for s in best_states:
                #     dist[s] = bestProb / len(best_states)
                # for s in node.successors:
                #     dist[s] += (1-bestProb) / len(node.successors)
                # dist.normalize()
                # # node = util.chooseFromDistribution(dist)
                # min_state = min(node.successors, key=lambda s: s.score)
                # node = min_state

                # Assuming ghost will choose action with minimum score
                # get the minimum score and states with the minimum score
                min_score = min(scores)
                best_indices = [index for index in range(len(scores)) if scores[index] == min_score]
                # Pick randomly among the best
                chosen_index = random.choice(best_indices)
                node = node.successors[chosen_index]
            is_max = not is_max

    def getAction(self, gameState):
        """
        Returns the minimax action
        """

        # if path is empty, generate tree and get path
        if not self.path:
            root = StateNode(Directions.STOP, float('-inf'), gameState)
            tree = self.generateMaxNode(float('-inf'), float('inf'), root, self.depth*2)

            self.generatePath(tree)
        # path is not empty so get next action
        return self.path.pop(0)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

