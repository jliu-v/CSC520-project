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
from operator import attrgetter


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


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    directionsList = ['West', 'East', 'North', 'South', 'Stop']
    legalMoves = currentGameState.getLegalActions()
    # print("Legal Moves:", legalMoves)

    scores = []
    for action in directionsList:
        if action in legalMoves:
            successorGameState = currentGameState.generatePacmanSuccessor(action)
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.path = []

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
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def generateMaxNode(self, alpha, beta, state, depth):
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

        # if reached depth limit, get min predicted score
        if depth == 0:
            predicted_scores = scoreEvaluationFunction(state.gameState)
            state.score = np.nanmin(predicted_scores)
            # print("depth limit action: ", state.action, " score: ", state.score)
            return state

        # get legal actions of current state
        for action in state.gameState.getLegalActions():
            new_state = StateNode(action, float('inf'), state.gameState.generatePacmanSuccessor(action))
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

        # if reached depth limit, get max predicted score
        if depth == 0:
            predicted_scores = scoreEvaluationFunction(state.gameState)
            state.score = np.nanmax(predicted_scores)
            # print("depth limit score ", state.score)
            return state

        # get legal actions of current state
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
        is_max = True

        while node.successors:
            if is_max:
                max_state = max(node.successors, key=lambda s: s.score)
                self.path.append(max_state.action)
                node = max_state
            else:
                min_state = min(node.successors, key=lambda s: s.score)
                node = min_state
            is_max = not is_max

    def getAction(self, gameState):
        """
        Returns the minimax action
        """
        print("path", self.path)
        # if path is empty, generate tree and get path
        if not self.path:
            root = StateNode(Directions.STOP, float('-inf'), gameState)
            tree = self.generateMaxNode(float('-inf'), float('inf'), root, self.depth*2)

            self.generatePath(tree)

        return self.path.pop(0)

    # def getAction(self, gameState):
    #     """
    #     Returns the minimax action
    #     """
    #     # Collect legal moves and successor states
    #     legal_moves = gameState.getLegalActions()
    #
    #     # Choose one of the best actions
    #     scores = [self.generateMinNode(float('-inf'), float('inf'), gameState.generateSuccessor(self.index, action), self.depth*2 - 1)
    #               for action in legal_moves]
    #     best_score = max(scores)
    #     best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
    #     chosen_index = random.choice(best_indices)  # Pick randomly among the best
    #
    #     # don't allow stop unless it's the only best move
    #     if legal_moves[chosen_index] == 'Stop' and len(best_indices) > 1:
    #         best_indices.remove(chosen_index)
    #         chosen_index = random.choice(best_indices)  # Randomly pick best without stop
    #
    #     return legal_moves[chosen_index]


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
