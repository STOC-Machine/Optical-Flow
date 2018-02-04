import pygame
import sys

def main():
    """
    A simple pygame program that creates a box and moves it across the screen
    Created to test optical flow tracking capabilites given known speed
    """
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((960,720))
    WHITE=(255,255,255)
    blue=(0,0,255)
    screen.fill(WHITE)
    x = 200
    y = 200
    pygame.draw.rect(screen,blue,(x,y,100,50),1)

    while True:
        msElapsed = clock.tick(30)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.fill(WHITE)
        x+=1
        pygame.draw.rect(screen,blue,(x,y,100,50),1)
        pygame.display.update()

main()
