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
        ghostDists = 0
        ghostWarning = 0
        for i in range(len(newGhostStates)):
            currDist = manhattanDistance(newPos, newGhostStates[i].getPosition())
            ghostDists += currDist
            
            if currDist < 3:
                ghostWarning += 1
        
        minFoodDist = -1
        for foodPos in newFood.asList():
            if (minFoodDist == -1):
                minFoodDist = manhattanDistance(newPos, foodPos)
            else:
                currFoodDist = manhattanDistance(newPos, foodPos)
                if (currFoodDist < minFoodDist):
                    minFoodDist = currFoodDist

        return successorGameState.getScore() + ghostDists / (minFoodDist * 5) - (50 * ghostWarning)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

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
        def minimax(gameState, agentIdx, depth):
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                    return self.evaluationFunction(gameState)
            
            if (agentIdx == 0):
                legalActions = gameState.getLegalActions(agentIdx)
                return max(minimax(gameState.generateSuccessor(agentIdx, action), 1, depth) for action in legalActions)
            else:
                nextAgentIdx = agentIdx + 1
                if (nextAgentIdx == gameState.getNumAgents()):
                    nextAgentIdx = 0
                    depth += 1
                legalActions = gameState.getLegalActions(agentIdx)
                return min(minimax(gameState.generateSuccessor(agentIdx, action), nextAgentIdx, depth) for action in legalActions)
        
        legalActions = gameState.getLegalActions(0)
        v = float('-inf')
        for action in legalActions:
            succGameState = gameState.generateSuccessor(0, action)
            v2 = minimax(succGameState, 1, 0)
            if v < v2:
                v = v2
                bestAction = action
        return bestAction          

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimaxABP(gameState, agentIdx, depth, alpha, beta):
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                    return self.evaluationFunction(gameState)
            
            if (agentIdx == 0):
                bestV = float('-inf')
                legalActions = gameState.getLegalActions(agentIdx)
                for action in legalActions:
                    v = minimaxABP(gameState.generateSuccessor(agentIdx, action), 1, depth, alpha, beta)
                    bestV = max(bestV, v)
                    alpha = max(alpha, bestV)
                    if (beta < alpha):
                        break
                return bestV
            else:
                nextAgentIdx = agentIdx + 1
                if (nextAgentIdx == gameState.getNumAgents()):
                    nextAgentIdx = 0
                    depth += 1
                bestV = float('inf')
                legalActions = gameState.getLegalActions(agentIdx)
                for action in legalActions:
                    v = minimaxABP(gameState.generateSuccessor(agentIdx, action), nextAgentIdx, depth, alpha, beta)
                    bestV = min(bestV, v)
                    beta = min(beta, bestV)
                    if (beta < alpha):
                        break
                return bestV
        
        legalActions = gameState.getLegalActions(0)
        v = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in legalActions:
            succGameState = gameState.generateSuccessor(0, action)
            v2 = minimaxABP(succGameState, 1, 0, alpha, beta)
            if v < v2:
                v = v2
                bestAction = action
            alpha = max(alpha, v)
        return bestAction    
        

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
        def expectimax(gameState, agentIdx, depth):
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                    return self.evaluationFunction(gameState)
            
            if (agentIdx == 0):
                legalActions = gameState.getLegalActions(agentIdx)
                return max(expectimax(gameState.generateSuccessor(agentIdx, action), 1, depth) for action in legalActions)
            else:
                nextAgentIdx = agentIdx + 1
                if (nextAgentIdx == gameState.getNumAgents()):
                    nextAgentIdx = 0
                    depth += 1
                legalActions = gameState.getLegalActions(agentIdx)
                return sum(expectimax(gameState.generateSuccessor(agentIdx, action), nextAgentIdx, depth) for action in legalActions)
        
        legalActions = gameState.getLegalActions(0)
        v = float('-inf')
        for action in legalActions:
            succGameState = gameState.generateSuccessor(0, action)
            v2 = expectimax(succGameState, 1, 0)
            if v < v2:
                v = v2
                bestAction = action
        return bestAction  

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    ghostDists = 0
    ghostWarning = 0
    for i in range(len(newGhostStates)):
        currDist = manhattanDistance(newPos, newGhostStates[i].getPosition())
        ghostDists += currDist
            
        if currDist < 2:
            ghostWarning += 1
        
    minFoodDist = -1
    foodSum = 0
    for foodPos in newFood.asList():
        foodSum += manhattanDistance(newPos, foodPos)
        """
        if (minFoodDist == -1):
            minFoodDist = manhattanDistance(newPos, foodPos)
        else:
            currFoodDist = manhattanDistance(newPos, foodPos)
            if (currFoodDist < minFoodDist):
                minFoodDist = currFoodDist
        """
    if (not len(newFood.asList()) == 0):
        foodAvg = foodSum / len(newFood.asList())
    else:
        foodAvg = 1

    return currentGameState.getScore() + (ghostDists / len(newGhostStates)) / (foodAvg * 10) - (50 * ghostWarning)

# Abbreviation
better = betterEvaluationFunction
