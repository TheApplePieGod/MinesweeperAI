import torch
import pygame
import random
import numpy as np
from board import Board
from collections import deque

BATCH_SIZE = 128
NUM_EPISODES = 1000000
MAX_STEPS = 1000
TRAIN_STEPS = 4

"""
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.set_default_device("mps")
"""
device = torch.device("cpu")

class Agent(torch.nn.Module):
    def __init__(self, input_size):
        super(Agent, self).__init__()

        self.input_size = input_size
        self.memory = deque(maxlen=BATCH_SIZE * 8)
        self.input_dim = 10
        self.embed_dim = 32
        self.num_heads = 16

        self.embed = torch.nn.Embedding(self.input_dim, self.embed_dim)
        self.attn = torch.nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            batch_first=True
        )
        self.flat = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.1)
        self.dense1 = torch.nn.Linear(self.embed_dim * self.input_size, 512)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(512, 256)
        self.relu = torch.nn.ReLU()
        self.dense3 = torch.nn.Linear(256, 1)
        self.loss = torch.nn.BCELoss()

        self.optimizer = torch.optim.Adam(self.parameters())

        self.load_state_dict(torch.load("weights/model3-v1"))

    def forward(self, x):
        embedding = self.embed(x)
        out, _ = self.attn(embedding, embedding, embedding)
        result = self.flat(out)
        result = self.dropout(result)
        result = self.dense1(result)
        result = self.relu(result)
        result = self.dense2(result)
        result = self.relu(result)
        result = self.dense3(result)
        return torch.sigmoid(result)

    def self_attention(x):
        n, d = x.shape
        d_q, d_k, d_v = 10, 10, 10

        Wq = torch.nn.Linear(d, d_q, bias=False)  # d_q x d
        Wk = torch.nn.Linear(d, d_k, bias=False)  # d_k x d
        Wv = torch.nn.Linear(d, d_v, bias=False)  # d_v x d

        Q = Wq(x)
        K = Wk(x)
        V = Wk(x)

        attn_mat = (Q @ K.T) / d_k ** 0.5
        attn_weights = torch.softmax(attn_mat, dim=0)

        context_vec = attn_weights @ V
        return context_vec

    def train_once(self):
        self.train()

        total_loss = 0
        total_steps = 0
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for i in range(1):
            s = torch.from_numpy(np.array([b[0] for b in minibatch])).to(device)
            t = torch.tensor([b[1] for b in minibatch])
            self.optimizer.zero_grad()
            output = self.forward(s)
            loss = self.loss(output, t)
            loss.backward()
            self.optimizer.step()

            total_loss += loss
            total_steps += 1

        print(f"Avg loss: {total_loss / total_steps}")

    def act(self, state, indices):
        with torch.no_grad():
            self.eval()

            # Predict mine probablities for each square 
            s = torch.from_numpy(state).to(device)
            probabilities = self.forward(s)

            # Select lowest probability
            selection = torch.argmin(probabilities)

            self.last_probabilities = probabilities

            action = indices[selection]
            return action, state[selection]

    def store(self, state, mine_chance):
        if np.all(state == 9):
            return
        self.memory.append((state, mine_chance))

    def save_weights(self):
        torch.save(self.state_dict(), "weights/model3-v1.5")


if __name__ == "__main__":
    board = Board(16, 16, 40)

    agent = Agent(board.get_observation_shape()[0])

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
        state, indices = board.get_observation()

        episode_reward = 0
        game_step = 0
        game_states = []
        for timestep in range(MAX_STEPS):
            game_step += 1

            # Select action
            action, acting_state = agent.act(state, indices)

            # Take action    
            terminated, is_loss = board.step(action)

            # Store action in memory
            # Skip first few plays because they are likely to be random
            if game_step >= 4:
                game_states.append((acting_state, [1.0 if is_loss else 0.0]))

            # Train every TRAIN_STEPS or if the game is over
            """
            if game_step % TRAIN_STEPS == 0 or terminated:
                if len(agent.memory) > BATCH_SIZE:
                    agent.train_once()
            """

            # Update visualization
            draw()

            if terminated:
                print('Episode {} end after n steps = {}'.format(e, game_step))
                if not is_loss > 0.0: # Win
                    if e % 2 == 0:
                        small_wins += 1
                    else:
                        med_wins += 1

                # Add game states to memory if valid
                if len(game_states) > 0:
                    for s in game_states:
                        agent.store(s[0], s[1])

                    if len(agent.memory) >= BATCH_SIZE:
                        agent.train_once()
                    else:
                        print(f"Memory size: {len(agent.memory)}")

                break

            state, indices = board.get_observation()

        # Average win rate over 10 games for each board type
        if (e + 1) % 20 == 0:
            small_win_rate = small_wins / 10.0
            med_win_rate = med_wins / 10.0
            small_wins = 0
            med_wins = 0

        # Write graph data
        if (e + 1) % 50 == 0:
            agent.save_weights()
