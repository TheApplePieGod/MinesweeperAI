from typing import Tuple
import pygame

def button(surface: pygame.Surface, pos: Tuple[int, int], size: Tuple[int, int], text: str, color: Tuple[int, int, int] = (255, 0, 0)) -> bool:
    pygame.draw.rect(surface, color, [pos[0], pos[1], size[0], size[1]])
    font = pygame.font.SysFont('Corbel',35)
    text = font.render(text, True, (255, 255, 255))
    surface.blit(text, pos)
    mouse = pygame.mouse.get_pos()
    pressed = pygame.mouse.get_pressed()[0]
    return pressed and mouse[0] >= pos[0] and mouse[0] <= pos[0] + size[0] and mouse[1] >= pos[1] and mouse[1] <= pos[1] + size[1]