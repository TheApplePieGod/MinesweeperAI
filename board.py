from math import trunc
from typing import List, Tuple
import numpy as np
import random
import pygame
import os
from button import button

GRID_SIZE = 35
SPACING_SIZE = 1
IMAGES = {}

# Load all images in the images folder
def load_images() -> None:
    paths = os.listdir("images")
    for path in paths:
        IMAGES[path] = pygame.transform.scale(
            pygame.image.load(f"images/{path}"),
            (GRID_SIZE, GRID_SIZE)
        )
load_images()

class Board:
    def __init__(self, width: int, height: int, mines: int) -> None:
        self.width = width
        self.height = height
        self.mines = mines
        self.score = 0
        self.flagged_mines = 0
        self.game_status = 0 # 0: Playing, 1: Win, 2: Loss
        self.first_click = True
        self.board = np.array([], np.ubyte)
        self.hidden = []
        self.show_mines = False
        self.show_nums = False
        self.observation_radius = 3

    def is_mine(self, index: int) -> bool:
        return self.board[index] & np.ubyte(9) == 9

    def set_mine(self, index: int) -> None:
        self.board[index] = np.ubyte(9)

    def has_flag(self, index: int) -> bool:
        return bool(self.board[index] & np.ubyte(32))

    def set_flag(self, index: int) -> None:
        if self.is_revealed(index): return
        if self.has_flag(index):
            self.board[index] &= ~np.ubyte(32)
            if self.is_mine(index):
                self.flagged_mines -= 1
        else:
            self.board[index] |= np.ubyte(32)
            if self.is_mine(index):
                self.flagged_mines += 1

        if self.flagged_mines == self.mines:
            self.game_status = 1
            self.score += 50

    def get_num(self, index: int) -> int:
        return self.board[index] & np.ubyte(15)

    def set_num(self, index: int, value: int) -> None:
        self.board[index] = np.ubyte(value)

    def is_revealed(self, index: int) -> bool:
        return bool(self.board[index] & np.ubyte(16))

    def reveal(self, index: int) -> None:
        if self.is_revealed(index): return
        if self.has_flag(index): return
        self.board[index] |= np.ubyte(16)

        self.hidden.remove(index)

        if self.is_mine(index):
            self.game_status = 2
            return

        self.score += 1

        # Reveal surrounding tiles
        current_num = self.get_num(index)
        surround = self.get_surrounding_indices(index)
        for i in surround:
            if current_num == 0: # Reveal all tiles around empties
                self.reveal(i)

        # Check for a win if all remaining tiles are mines
        for h in self.hidden:
            if not self.is_mine(h):
                return

        self.game_status = 1

    def get_surrounding_indices(self, index: int) -> List[int]:
        indices = []
        x = (index % self.width)
        y = (index // self.width)
        up = y > 0
        left = x > 0
        down = y < self.height - 1
        right = x < self.width - 1
        if up:
            indices.append(index - self.width)
            if left:
                indices.append(index - self.width - 1)
            if right:
                indices.append(index - self.width + 1)
        if down:
            indices.append(index + self.width)
            if left:
                indices.append(index + self.width - 1)
            if right:
                indices.append(index + self.width + 1)
        if left:
            indices.append(index - 1)
        if right:
            indices.append(index + 1)
        return indices

    def is_guess(self, index: int) -> bool:
        for i in self.get_surrounding_indices(index):
            if self.is_revealed(i):
                return False
        return True

    def get_score(self) -> int:
        return self.score

    # Get the observational value of a given tile
    def get_tile_observation(self, index: int):
        if not self.is_revealed(index):
            final = 9
        else:
            final = self.get_num(index) # 0-8
        return final

    # Get observational values of all tiles in a given radius
    def get_radius_observation(self, index: int):
        radius = self.observation_radius
        size = radius * 2 + 1
        mat = np.zeros((size * size), dtype=np.int32)
        cur_x = (index % self.width)
        cur_y = (index // self.width)
        for y in range(0, size):
            for x in range(0, size):
                new_x = (x - radius) + cur_x
                new_y = (y - radius) + cur_y
                if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
                    mat[x + y * size] = 9 # No information == 9
                    continue
                new_index = new_x + (new_y * self.width)
                mat[x + y * size] = self.get_tile_observation(new_index)
        return mat

    # Gets radius observation of all tiles on the board that are not revealed 
    def get_observation(self):
        observations = []
        indices = []
        for i in range(len(self.board)):
            if not self.is_revealed(i):
                observations.append(self.get_radius_observation(i))
                indices.append(i)
        return np.array(observations), np.array(indices)

    # Gets radius observation of all tiles on the board that are mines
    def get_mine_observation(self):
        observations = []
        indices = []
        for i in range(len(self.board)):
            if self.is_mine(i) and not self.is_revealed(i):
                observations.append(self.get_radius_observation(i))
                indices.append(i)
        return np.array(observations), np.array(indices)

    def get_observation_shape(self):
        radius = self.observation_radius
        size = radius * 2 + 1
        return (size * size,)

    # Each square can either reveal
    def get_action_size(self):
        return self.width * self.height * 1

    def get_random_action(self):
        return random.sample(self.get_valid_actions(), 1)[0]

    def decode_action(self, action_val: int):
        square = action_val
        return square

    def get_valid_actions(self) -> List[int]:
        valid = []
        for i in range(self.get_action_size()):
            if self.is_action_valid(i):
                valid.append(i)
        return valid

    def is_action_valid(self, action_val: int):
        square = self.decode_action(action_val)
        if self.is_revealed(square) or self.has_flag(square):
            return False
        return True

    # Update the board given an action and return whether or not
    # the game has ended and whether or not the termination was a loss
    def step(self, action_val: int):
        square = self.decode_action(action_val)

        self.handle_first_click(square)

        if not self.is_action_valid(action_val):
            self.game_status = 2
        else:
            self.reveal(square)

        # terminated, isLoss
        return self.game_status != 0, self.game_status == 2

    # Reset the board and generate mines
    def generate(self) -> None:
        self.board = np.array([np.ubyte(0)] * self.width * self.height, np.ubyte)
        self.hidden = [i for i in range(len(self.board))]
        self.score = 0
        self.flagged_mines = 0
        self.first_click = True
        self.game_status = 0
        mines_left = self.mines
        while mines_left > 0:
            loc = random.randrange(0, len(self.board))
            if not self.is_mine(loc):
                self.set_mine(loc)
                mines_left -= 1
        return self.get_observation()

    # Populate empty cells with numbers of surrounding mines
    def populate_numbers(self) -> None:
        for i in range(len(self.board)):
            if self.is_mine(i): continue
            mine_count = 0
            surround = self.get_surrounding_indices(i)
            for s in surround:
                mine_count += self.is_mine(s)
            self.set_num(i, mine_count)

    def get_board_index_from_mouse(self, mouse_pos: Tuple[int, int]) -> int:
        x = trunc(mouse_pos[0] / (GRID_SIZE + SPACING_SIZE))
        y = trunc(mouse_pos[1] / (GRID_SIZE + SPACING_SIZE))
        if x >= self.width or y >= self.height:
            return -1
        return x + y * self.width

    # Converts a mouse position to a board index and clicks
    def click_pos(self, mouse_pos: Tuple[int, int], button: int) -> None:
        index = self.get_board_index_from_mouse(mouse_pos)
        if index == -1:
            return
        self.click(index, button)

    # Must be called before board actions can take place
    def handle_first_click(self, index: int) -> None:
        if self.first_click:
            if self.is_mine(index): # Move the mine somewhere else
                i = -1
                while i == -1 or self.is_mine(i):
                    i = random.randrange(0, len(self.board))
                # while self.is_mine(i := random.randrange(0, len(self.board))):
                #    pass
                self.set_num(index, 0)
                self.set_mine(i)

            self.populate_numbers()
            self.first_click = False

    # Handle a player click on an index
    def click(self, index: int, button: int) -> None:
        if self.game_status != 0: return
        if index < 0 or index >= len(self.board): return

        self.handle_first_click(index)

        if button == 1: # Left click
            self.reveal(index)
        elif button == 3: # Right click
            self.set_flag(index)

    def get_tile_coord(self, index) -> (int, int):
        x = (index % self.width) * (GRID_SIZE + SPACING_SIZE) 
        y = (index // self.width) * (GRID_SIZE + SPACING_SIZE)
        return x, y

    def draw_transparent_rect(self, surface, color, rect):
        shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
        surface.blit(shape_surf, rect)

    def draw_board(self, surface: pygame.Surface) -> None:
        for i in range(len(self.board)):
            x, y = self.get_tile_coord(i)
            num = self.get_num(i)

            # Revealed base
            surface.blit(IMAGES["tile_revealed.png"], (x, y))

            if num == 0: # Empty
                if not self.is_revealed(i) and not self.show_nums:
                    surface.blit(IMAGES["tile_hidden.png"], (x, y))
            elif num > 0 and num < 9: # Some number of mines 1-8
                if self.is_revealed(i) or self.show_nums:
                    surface.blit(IMAGES[f"number_{num}.png"], (x, y))
                else:
                    surface.blit(IMAGES["tile_hidden.png"], (x, y))
            elif num == 9: # Mine
                if self.is_revealed(i) or self.show_mines:
                    pygame.draw.rect(surface, (255, 0, 0), [x, y, GRID_SIZE, GRID_SIZE])
                    surface.blit(IMAGES["mine.png"], (x, y))
                else:
                    surface.blit(IMAGES["tile_hidden.png"], (x, y))

            if self.has_flag(i): # Flag
                surface.blit(IMAGES["flag.png"], (x, y))

        if self.game_status == 1:
            font = pygame.font.SysFont('Corbel', 60, True)
            text = font.render("WIN", True, (0, 255, 0))
            surface.blit(text, (0, 0))
        elif self.game_status == 2:
            font = pygame.font.SysFont('Corbel', 60, True)
            text = font.render("GAME OVER", True, (255, 0, 0))
            surface.blit(text, (0, 0))

    def draw_attention_weights(self, surface: pygame.Surface, index, weights):
        radius = self.observation_radius
        size = radius * 2 + 1

        # Extract & normalize attention with respect to the center tile
        weights = weights[radius]
        weights /= weights.max()

        cur_x = (index % self.width)
        cur_y = (index // self.width)
        for y in range(0, size):
            for x in range(0, size):
                new_x = (x - radius) + cur_x
                new_y = (y - radius) + cur_y
                if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
                    continue
                weight = weights[y * size + x]
                new_index = new_x + (new_y * self.width)
                tile_x, tile_y = self.get_tile_coord(new_index)
                self.draw_transparent_rect(
                    surface,
                    (255, 0, 0, weight * 255),
                    pygame.Rect(tile_x, tile_y, GRID_SIZE, GRID_SIZE)
                )

    def draw_mine_chance(self, surface: pygame.Surface, chances, indices) -> None:
        font = pygame.font.SysFont('Corbel',12)
        for i, chance in enumerate(chances):
            x, y = self.get_tile_coord(indices[i])
            text = font.render(f"{chance[0] * 100:0.2f}%", True, (255, 0, 0))
            surface.blit(text, (x, y + GRID_SIZE / 2))

    def draw_other(self, surface: pygame.Surface) -> None:
        y = self.height * (GRID_SIZE + SPACING_SIZE)
        bwidth = 100
        nextpos = bwidth + 5
        if button(surface, (0, y), (bwidth, 30), "Gen"):
            self.generate()
        if button(surface, (nextpos, y), (bwidth, 30), "SMin"):
            self.show_mines = True
        if button(surface, (nextpos * 2, y), (bwidth, 30), "HMin"):
            self.show_mines = False
        if button(surface, (nextpos * 3, y), (bwidth, 30), "SNum"):
            self.show_nums = True
        if button(surface, (nextpos * 4, y), (bwidth, 30), "HNum"):
            self.show_nums = False
