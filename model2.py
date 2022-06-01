import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
import random
import pygame
from board import Board
from collections import deque
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.losses import Huber, BinaryCrossentropy
import datetime as dt

BATCH_SIZE = 512
NUM_EPISODES = 1000000
MAX_STEPS = 1000
TRAIN_STEPS = 4

class Agent:
    def __init__(self, state_size, trainable):
        self.state_size = state_size
        self.optimizer = Adam(learning_rate=0.001)
        
        self.memory = deque(maxlen=5_000)
                
        self.network = self.build_model(trainable)
        self.last_probabilities = []

    def store(self, state, mine_chance):
        self.memory.append((state, mine_chance))
    
    def build_model(self, trainable):
        model = Sequential()
        model.add(Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=HeUniform(), input_shape=self.state_size))
        model.add(MaxPooling2D((2, 2), padding="same"))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer=HeUniform()))
        model.add(MaxPooling2D((2, 2), padding="same"))
        model.add(Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer=HeUniform()))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer=HeUniform()))
        model.add(Dense(512, activation='relu', kernel_initializer=HeUniform()))
        model.add(Dense(1, activation="sigmoid"))

        if trainable:
            model.compile(loss=BinaryCrossentropy(), optimizer=self.optimizer, metrics=['binary_accuracy'])
        else:
            model.trainable = False

        try:
            model.load_weights('checkpoints/model2_checkpoint')
            print("Loaded model weights")
        except:
            pass

        return model

    def save_weights(self):
        self.network.save_weights("checkpoints/model2_checkpoint")
    
    def act(self, state, indices):
        # Predict mine probablities for each square 
        probabilities = self.network.predict(state, verbose=0)
        
        # Select lowest probability
        selection = np.argmin(probabilities)

        self.last_probabilities = probabilities

        action = indices[selection]
        return action, state[selection]

    def train(self):
        # Select random batch of states from memory
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([mem[0] for mem in minibatch])
        probabilities = np.array([mem[1] for mem in minibatch])

        # Train from the random batch
        history = self.network.fit(states, probabilities, verbose=0, shuffle=True, batch_size=BATCH_SIZE)
        return history

# https://rubikscode.net/2021/07/13/deep-q-learning-with-python-and-tensorflow-2-0/
# https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/
# https://github.com/mswang12/minDQN/blob/main/minDQN.py
# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
if __name__ == "__main__":
    board = Board(16, 16, 40)
    time_str = dt.datetime.now().strftime('%d%m%Y%H%M')
    train_writer = tf.summary.create_file_writer(f"summaries/Sweeper_{time_str}")

    agent = Agent(board.get_observation2_shape(), True)

    agent.network.summary()

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

    small_win_rate = None
    small_wins = 0
    med_win_rate = None
    med_wins = 0
    loss = None
    accuracy = None
    for e in range(0, NUM_EPISODES):
        # Reset the enviroment
        # Switch off training a small and medium board
        if e % 2 == 0:
            board = Board(9, 9, 10)
        else:
            board = Board(16, 16, 40)
        board.generate()
        state, indices = board.get_observation2()

        episode_reward = 0
        game_step = 0
        for timestep in range(MAX_STEPS):
            game_step += 1

            # Select action
            action, acting_state = agent.act(state, indices)
            
            # Take action    
            terminated, is_loss = board.step2(action)

            # Store action in memory
            agent.store(acting_state, 1.0 if is_loss else 0.0)
            
            # Train every TRAIN_STEPS or if the game is over
            if game_step % TRAIN_STEPS == 0 or terminated:
                if len(agent.memory) > BATCH_SIZE:
                    history = agent.train()
                    loss = history.history["loss"][0]
                    accuracy = history.history["binary_accuracy"][0]
            
            # Update visualization
            draw()

            if terminated:
                print('Episode {} end after n steps = {}'.format(e, game_step))
                if not is_loss > 0.0: # Win
                    if e % 2 == 0:
                        small_wins += 1
                    else:
                        med_wins += 1
                break
            
            state, indices = board.get_observation2()

        # Average win rate over 10 games for each board type
        if (e + 1) % 20 == 0:
            small_win_rate = small_wins / 10.0
            med_win_rate = med_wins / 10.0
            small_wins = 0
            med_wins = 0

        # Write graph data
        with train_writer.as_default():
            if accuracy != None:
                tf.summary.scalar('accuracy', accuracy, e)
            if loss != None:
                tf.summary.scalar('loss', loss, e)
            if small_win_rate != None:
                tf.summary.scalar('small_win_rate', small_win_rate, e)
            if med_win_rate != None:
                tf.summary.scalar('med_win_rate', med_win_rate, e)
            tf.summary.scalar('small_steps' if e % 2 == 0 else 'med_steps', game_step, e)
                
        if (e + 1) % 50 == 0:
            agent.save_weights()
