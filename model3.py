import math
import torch
import pygame
import random
import numpy as np
import datetime as dt
from torchinfo import summary
from board import Board
from collections import deque
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 256
NUM_EPISODES = 1000000
MAX_STEPS = 1000
EXPERIMENT_NAME = "model3-v5"

"""
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.set_default_device("mps")
"""
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_device("cuda")
else:
    device = torch.device("cpu")

class PositionalEncoding(torch.nn.Module):
    def __init__(self, embed_dim, width, height):
        super(PositionalEncoding, self).__init__()

        if embed_dim % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(embed_dim))
        pe = torch.zeros(embed_dim, height, width)
        # Each dimension use half of d_model
        embed_dim = int(embed_dim / 2)
        div_term = torch.exp(torch.arange(0., embed_dim, 2) *
                             -(math.log(10000.0) / embed_dim))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:embed_dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:embed_dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[embed_dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[embed_dim + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe = pe.reshape((embed_dim * 2, height * width)).T.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dense_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )
        self.dense1 = torch.nn.Linear(embed_dim, dense_dim)
        self.dense2 = torch.nn.Linear(dense_dim, embed_dim)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.dense1(x)
        ff_output = self.relu(ff_output)
        ff_output = self.dense2(ff_output)
        x = self.norm2(x + self.dropout(ff_output))
        return x

    def get_attention_weights(self, x, average):
        _, weights = self.self_attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=average
        )
        return weights

class Agent(torch.nn.Module):
    def __init__(self, input_size):
        super(Agent, self).__init__()

        self.input_size = input_size
        self.memory = deque(maxlen=BATCH_SIZE * 8)
        self.num_layers = 4
        self.input_dim = 11
        self.embed_dim = 64
        self.dense_dim = 512
        self.dropout = 0.1
        self.num_heads = 16
        self.last_probabilities = []
        self.train_steps = 0

        self.state_embed = torch.nn.Embedding(self.input_dim, self.embed_dim)
        self.pos_embed = PositionalEncoding(
            self.embed_dim,
            int(self.input_size ** 0.5),
            int(self.input_size ** 0.5)
        )
        self.norm = torch.nn.LayerNorm(self.embed_dim)
        self.layers = torch.nn.ModuleList([
            EncoderLayer(self.embed_dim, self.num_heads, self.dense_dim, self.dropout)
            for i in range(self.num_layers)
        ])
        self.flat = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(self.embed_dim * self.input_size, self.dense_dim)
        self.dense2 = torch.nn.Linear(self.dense_dim, self.dense_dim)
        self.dense3 = torch.nn.Linear(self.dense_dim, 1)
        self.relu = torch.nn.ReLU()
        self.loss = torch.nn.BCELoss()

        self.optimizer = torch.optim.Adam(self.parameters())

        summary(self, input_data=torch.zeros(1, self.input_size, dtype=torch.int))

        # raise ValueError

        try:
            self.load_state_dict(
                torch.load(f"weights/{EXPERIMENT_NAME}", map_location=device)
            )
        except:
            pass

    def embed(self, x):
        state_embed = self.state_embed(x)
        pos_embed = self.pos_embed(state_embed)
        return self.norm(pos_embed)

    def forward(self, x):
        result = self.embed(x)
        for layer in self.layers:
            result = layer(result)
        result = self.flat(result)
        result = self.dense1(result)
        result = self.relu(result)
        result = self.dense2(result)
        result = self.relu(result)
        result = self.dense3(result)
        return torch.sigmoid(result)

    def get_attention_weights(self, x, average):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).unsqueeze(0).to(device)
        x = self.embed(x)

        weights = torch.zeros((1, self.input_size, self.input_size))
        with torch.no_grad():
            self.eval()
            for layer in self.layers:
                weights += layer.get_attention_weights(x, average)

        return weights[0]

    def train_once(self):
        self.train()

        total_loss = 0
        total_acc = 0
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
            total_acc += torch.sum(torch.round(output) == t) / output.shape[0]
            total_steps += 1

        self.train_steps += total_steps

        return total_loss / total_steps, total_acc / total_steps

    def act(self, state, indices):
        # Predict mine probablities for each square
        probabilities = self.get_probabilities(state)

        # Select lowest probability
        selection = torch.argmin(probabilities)

        self.last_probabilities = probabilities

        action = indices[selection]
        return action, state[selection]

    def get_probabilities(self, state):
        with torch.no_grad():
            self.eval()
            if len(state.shape) == 2:
                s = torch.from_numpy(state).to(device)
            else:
                s = torch.from_numpy(np.array([state])).to(device)
            return self.forward(s)

    def store(self, state, mine_chance):
        if np.all(state == 9):
            return
        self.memory.append((state, mine_chance))

    def save_weights(self):
        torch.save(self.state_dict(), f"weights/{EXPERIMENT_NAME}")


if __name__ == "__main__":
    board = Board(16, 16, 40)
    time_str = dt.datetime.now().strftime('%d%m%Y%H%M')
    train_writer = SummaryWriter(f"summaries/{EXPERIMENT_NAME}_{time_str}")
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
                    # Add in mine observations to match state cnount
                    for s in board.get_mine_observation()[0]:
                        # Ensure state is not primarily unrevealed
                        if np.count_nonzero(s == 9) / s.shape[0] <= 0.5:
                            agent.store(s, [1.0])

                    if len(agent.memory) >= BATCH_SIZE:
                        loss, accuracy = agent.train_once()
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

        if accuracy is not None:
            train_writer.add_scalar('accuracy', accuracy, e)
        if loss is not None:
            train_writer.add_scalar('loss', loss, e)
        if small_win_rate is not None:
            train_writer.add_scalar('small_win_rate', small_win_rate, e)
        if med_win_rate is not None:
            train_writer.add_scalar('med_win_rate', med_win_rate, e)
        train_writer.add_scalar('small_steps' if e % 2 == 0 else 'med_steps', game_step, e)
        train_writer.flush()

        # Write graph data
        if (e + 1) % 50 == 0:
            agent.save_weights()
