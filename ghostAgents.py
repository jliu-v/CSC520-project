# ghostAgents.py
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


from game import Agent
from game import Actions
from game import Directions
import random
import numpy as np
from util import manhattanDistance
import util


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared:
            speed = 0.5

        actionVectors = [Actions.directionToVector(
            a, speed) for a in legalActions]
        newPositions = [(pos[0]+a[0], pos[1]+a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(
            pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(
            legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1-bestProb) / len(legalActions)
        dist.normalize()
        return dist

def scoreEvaluationFunction(currentGameState, agentIndex):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    directionsList = ['West', 'East', 'North', 'South']
    legalMoves = currentGameState.getLegalActions(agentIndex)

    scores = []
    for action in directionsList:
        if action in legalMoves:
            successorGameState = currentGameState.generateSuccessor(agentIndex, action)
            scores.append(successorGameState.getScore())
        else:
            scores.append(np.nan)

    return scores


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
        self.index = 1
        # self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.path = []


class AlphaBetaGhostAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning
    """

    def generateMaxNode(self, alpha, beta, state, depth):

        # if current state is win or lose, return game state score
        if state.gameState.isWin() or state.gameState.isLose():
            state.score = state.gameState.getScore()
            return state

        # if reached depth limit, get max predicted score
        if depth == 0:
            predicted_scores = scoreEvaluationFunction(state.gameState, 0)
            state.score = np.nanmax(predicted_scores)
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
        return state

    def generateMinNode(self, alpha, beta, state, depth):

        # if current state is win or lose, return game state score
        if state.gameState.isWin() or state.gameState.isLose():
            state.score = state.gameState.getScore()
            return state

        # if reached depth limit, get min predicted score
        if depth == 0:
            predicted_scores = scoreEvaluationFunction(state.gameState, 1)
            state.score = np.nanmin(predicted_scores)
            # print("depth limit score ", state.score)
            return state

        # get legal actions of current state
        for action in state.gameState.getLegalActions(1):
            new_state = StateNode(action, float('-inf'), state.gameState.generateSuccessor(self.index, action))
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
        is_max = False

        while node.successors:
            scores = [s.score for s in node.successors]
            print("ghost:", scores)
            if is_max:
                max_state = max(node.successors, key=lambda s: s.score)
                node = max_state
            else:
                min_state = min(node.successors, key=lambda s: s.score)
                self.path.append(min_state.action)
                node = min_state
                # min_score = min(scores)
                # best_indices = [index for index in range(len(scores)) if scores[index] == min_score]
                # chosen_index = random.choice(best_indices)  # Pick randomly among the best
                # node = node.successors[chosen_index]
                # self.path.append(node.action)
            is_max = not is_max

    def getAction(self, gameState):
        """
        Returns the minimax action
        """
        print("ghost:")
        print("path", self.path)
        # if path is empty, generate tree and get path
        if not self.path:
            root = StateNode(Directions.STOP, float('-inf'), gameState)
            tree = self.generateMinNode(float('inf'), float('-inf'), root, self.depth*2 - 1)

            self.generatePath(tree)

        return self.path.pop(0)