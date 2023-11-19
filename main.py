import pygame, sys, time
from board import Board
from model3 import Agent

# Initialize pygame
pygame.init()

# Create the window that we will be drawing to
DISPLAY = pygame.display.set_mode((1200, 800), 0, 32)

# Initialize the board
#board = Board(9, 9, 10)
board = Board(16, 16, 40)
#board = Board(30, 16, 99)
board.generate()

# Initialize AI agent
agent = Agent(board.get_observation_shape()[0])

def make_ai_move():
    state, indices = board.get_observation()
    action, _ = agent.act(state, indices)
    return board.step(action)

def main():
    block_selection = -1
    autoplay = False
    while True:
        step_turn = autoplay

        # Check to see if we should quit the program
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                #board.click_pos(event.pos, event.button)
                block_selection = board.get_board_index_from_mouse(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    step_turn = True
                if event.key == pygame.K_RETURN:
                    autoplay = not autoplay

        # Clear the window by filling it with white pixels
        DISPLAY.fill((255, 255, 255))

        # Draw the board
        state, indices = board.get_observation()
        probs = agent.get_probabilities(state)
        board.draw_board(DISPLAY)
        if block_selection != -1:
            for i, idx in enumerate(indices):
                if idx == block_selection:
                    board.draw_attention_weights(
                        DISPLAY,
                        idx,
                        agent.get_attention_weights(state[i], True),
                    )
                    break
        board.draw_mine_chance(DISPLAY, probs, indices)
        board.draw_other(DISPLAY)

        # Tell pygame to present the updated window to the user
        pygame.display.update()

        if step_turn:
            time.sleep(0.2)
            if board.game_status != 0:
                board.generate()
                time.sleep(0.5)
            terminated, is_loss = make_ai_move()
            if is_loss:
                autoplay = False

# Start the main loop
main() 
