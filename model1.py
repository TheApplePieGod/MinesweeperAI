import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
import random
from board import Board
from collections import deque
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
import pygame
import datetime as dt

board = Board(9, 9, 10)

BATCH_SIZE = 32
NUM_EPISODES = 100000
NUM_TIMESTEPS = 1000

train_writer = tf.summary.create_file_writer(f"summaries/Sweeper_{dt.datetime.now().strftime('%d%m%Y%H%M')}")

class Agent:
    def __init__(self, optimizer):
        # Initialize atributes
        self.state_size = board.get_observation_shape()
        self.action_size = board.get_action_size()
        self.optimizer = optimizer
        
        self.memory = deque(maxlen=10000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.95
        self.epsilon = 0.95
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.sync_models()

    def store(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        model = Sequential()
        model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=self.state_size))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self.optimizer)

        try:
            model.load_weights('checkpoints/model1_checkpoint')
            print("Loaded model weights")
        except:
            pass

        return model

    # Sync the weights of the target and current networks
    def sync_models(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return board.get_random_action()
        
        # Predict q values for this state
        q_values = self.q_network.predict(state, verbose=0)

        # Purge invalid moves
        for index, value in np.ndenumerate(q_values):
            board_index = index[1]
            if not board.is_action_valid(board_index):
                q_values[index] = np.min(q_values)

        # Select move with highest q value
        action = np.argmax(q_values)
        return action

    def retrain(self):
        # Select random batch
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        for state, action, reward, next_state, terminated in minibatch:
            target = self.q_network.predict(state, verbose=0, batch_size=BATCH_SIZE)
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state, verbose=0, batch_size=BATCH_SIZE)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(state, target, epochs=1, verbose=0, shuffle=False, batch_size=BATCH_SIZE)

        # Decay epsilon (exploration rate)
        self.epsilon = max(0.01, self.epsilon * 0.99975)

# https://rubikscode.net/2021/07/13/deep-q-learning-with-python-and-tensorflow-2-0/
# https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/
optimizer = Adam(learning_rate=0.01)
agent = Agent(optimizer)

agent.q_network.summary()

# Initialize pygame
pygame.init()

# Create the window that we will be drawing to
window = pygame.display.set_mode((1000, 800), 0, 32)

def draw():
    # Wait for events
    pygame.event.get()

    # Clear the window by filling it with white pixels
    window.fill((255, 255, 255))

    # Draw the board
    board.draw_board(window)

    # Tell pygame to present the updated window to the user
    pygame.display.update()

for e in range(0, NUM_EPISODES):
    # Reset the enviroment
    board.generate()
    state = board.get_observation()
    state = np.reshape(state, [1, *board.get_observation_shape()])

    episode_reward = 0

    for timestep in range(NUM_TIMESTEPS):
        # Select action
        action = agent.act(state)
        
        # Take action    
        next_state, reward, terminated = board.step(action)
        next_state = np.reshape(next_state, [1, *board.get_observation_shape()])

        # Store memory
        agent.store(state, action, reward, next_state, terminated)
        
        state = next_state
        episode_reward += reward
        
        # Update visualization
        draw()

        if terminated:
            agent.sync_models()
            break
            
        if len(agent.memory) > BATCH_SIZE:
           agent.retrain()

    with train_writer.as_default():
        tf.summary.scalar('rewards', episode_reward, e)
            
    if (e + 1) % 10 == 0:
        print("**********************************")
        print("Episode: {}".format(e + 1))
        print("**********************************")
        agent.q_network.save_weights("checkpoints/model1_checkpoint")
