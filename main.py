import pygame, sys, time
from board import Board
from model3 import Agent

# Initialize pygame
pygame.init()

# Create the window that we will be drawing to
DISPLAY = pygame.display.set_mode((1000, 800), 0, 32)

# Initialize the board
board = Board(9, 9, 10)
board.generate()

# Initialize AI agent
agent = Agent(board.get_observation2_shape(), False)

def make_ai_move():
    state, indices = board.get_observation2()
    action, _ = agent.act(state, indices)
    board.step2(action)

def main():
    while True:
        # Check to see if we should quit the program
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                board.click_pos(event.pos, event.button)
 
        # Clear the window by filling it with white pixels
        DISPLAY.fill((255, 255, 255))
 
        # Draw the board
        board.draw_board(DISPLAY)
        # board.draw_other(DISPLAY)

        # Tell pygame to present the updated window to the user
        pygame.display.update()

        time.sleep(0.2)
        if board.game_status != 0:
            board.generate()
            time.sleep(0.5)
        make_ai_move()
 
# Start the main loop
main() 
