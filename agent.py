# A agent that plays the game othello against another agent or a human player using the command line.
# The agent should be able to play the game using a neural network that is trained using reinforcement learning.

from game import othelloGame
from collections import deque
import torch
import random
import numpy as np

from model import LinearQNet, QTrainer
from plot import plot

# Define constants
MAX_MEMORY = 100_000        # Maximum number of previous experiences we are storing
BATCH_SIZE = 1000           # Number of experiences we use for training per step

LEARING_RATE = 0.001        # How fast the neural network learns
EPSILON_CONSTANT = 80       # How many iterations we want to take before we reach our minimum epsilon

INPUT_SIZE = 1              # Number of inputs to the neural network
HIDDEN_SIZE = 22            # Number of neurons in the hidden layer
OUTPUT_SIZE = 2             # Number of outputs to the neural network

board = othelloGame().board
is_valid_move = othelloGame().is_valid_move
ai_make_move = othelloGame().ai_make_move
current_player = othelloGame().current_player



class Agent:
    def __init__(self, player) -> None:
        self.board = othelloGame().get_state()
        self.player = player
        self.depth = 4
        self.alpha = -float("inf")
        self.beta = float("inf")
        self.maximizing = True
        self.states = deque(maxlen=10)
        self.states.append(self.board)

        self.n_games = 0 # number of games we've played
        self.epsilon = 0 # exploration rate
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # long term memory

        self.model = LinearQNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, lr=LEARING_RATE, gamma=self.gamma)
        self.last_10_scores = [0] * 10  


    def get_state(self):
        # get the state of the board

        self.board[0][0] = 0

        state = float(np.array(self.board, dtype=np.float32))
        state = torch.from_numpy(state)
        state = state.unsqueeze(0).float()
        return state
    
    def get_valid_moves(self):
        # check for valid moves from the othelloGame().get_valid_moves() function
        valid_moves = []
        for i in range(8):
            for j in range(8):
                if is_valid_move(i, j, self.player):
                    valid_moves.append((i, j))
        return valid_moves
    
    def get_action(self, state):
        random_number = random.random()
        if random_number < self.epsilon:
            valid_moves = self.get_valid_moves()
            random_move = random.choice(valid_moves)
            return random_move
        else:
            prediction = self.model(state)
            probabilities = torch.nn.functional.softmax(prediction, dim=-1)
            action = torch.argmax(probabilities).item()
            return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        

def train():
    plot_scores = []
    plot_mean = []
    plot_last_10 = []

    total_score = 0
    record = 0
    
    player1 = 1
    player2 = -1

    # should have 2 agents, one for each player
    agent1 = Agent(player=player1)
    agent2 = Agent(player=player2)
    game = othelloGame()

    # play the game with the two agents taking turns making moves until the game is over
    while not game.is_game_over():
        # Get old state
        state_old = None

        # Get current player
        current_player = game.current_player

        action = [0, 0]

        agent = agent1

        # If it is agent 1's turn
        if current_player == player1:
            agent = agent1
            state_old = agent.get_state()
            valid_moves = agent.get_valid_moves()
            action = agent.get_action(state_old)
        # If it is agent 2's turn
        elif current_player == player2:
            agent = agent2
            state_old = agent.get_state()
            valid_moves = agent.get_valid_moves()
            action = agent.get_action(state_old)

        print(f"Player {current_player}: {action}")

        reward, something, score =game.ai_make_move(action, current_player)

        agent.states.append(game.board)
        state_new = agent.get_state()
        done = game.is_game_over()
        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)
        agent.train_long_memory()
        total_score += reward
        if done:
            agent.n_games += 1
            agent.last_10_scores.append(total_score)
            mean_score = sum(agent.last_10_scores) / len(agent.last_10_scores)
            plot_scores.append(total_score)
            plot_mean.append(mean_score)
            plot_last_10.append(agent.last_10_scores[-1])
            if total_score > record:
                record = total_score
            print(f"Player {current_player}: Total score: {total_score}, Mean score: {mean_score}")
        
    # Plot the results
    plot(plot_scores, plot_mean, plot_last_10)
