import math
from model import Model
from layer import *
import numpy as np
import pygame
import pickle

def visualizeModel(m, surface, inp):
    res = m.predict(inp)
    pygame.draw.rect(surface, background, pygame.Rect(30, 330, 700, 300))
    for r in range(10):
        pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(int(60+60*r), 630-int(res[0][r]*300), 30, int(res[0][r]*300)))
    pygame.draw.rect(surface, background, pygame.Rect(320, 0, 700, 300))
    text = fontbig.render(str(np.argmax(res)), True, (255, 255, 255))
    screen.blit(text, (600, 120))
    text = font2.render(str(np.around(res[0][np.argmax(res)]*100, 1))+'% confidence', True, (255, 255, 255))
    screen.blit(text, (575, 250))
    text = fontbig.render('>>', True, (255, 255, 255))
    screen.blit(text, (400, 120))
    pygame.display.flip()


file = open('512x128x128x128x64', 'rb')
model = pickle.load(file)
file.close()

points = []
grid = [[0]*28 for i in range(28)]
toInput = lambda x: np.asarray(x).T.reshape(1, 784)

pygame.init()
font = pygame.font.Font('cmunbx.ttf', 32)
fontbig = pygame.font.Font('cmunbx.ttf', 96)
font2 = pygame.font.Font('cmunbx.ttf', 14)
background = (19, 20, 23)
screen = pygame.display.set_mode((720, 720))
screen.fill(background) 
for i in range(10):
    text = font.render(str(i), True, (255, 255, 255))
    screen.blit(text, (70+60*i, 640))
text = fontbig.render('>>', True, (255, 255, 255))
screen.blit(text, (400, 120))

visualizeModel(model, screen, toInput(grid))
pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(20, 20, 300, 300), 1)
pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(30, 30, 280, 280), 1)
pygame.display.flip()

def distanceVal(p, x, y):
    return max(0, math.sqrt((((p[0]-30)-x*10))**2+(((p[1]-30)-y*10))**2)*-0.05+1)

running = True
  
while running:
    
    for event in pygame.event.get():
        if event.type == pygame.NOEVENT:
            continue
        if pygame.mouse.get_pressed()[0]:
            p = pygame.mouse.get_pos()
            if p[0] > 30 and p[0] < 310 and p[1] > 30 and p[1] < 310:
                x, y = (p[0]-30)//10, (p[1]-30)//10
                vals = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
                for v in vals:
                    grid[max(min(27, x+v[0]), 0)][max(min(27, y+v[1]), 0)] = max(grid[max(min(27, x+v[0]), 0)][max(min(27, y+v[1]), 0)], distanceVal(p, max(min(27, x+v[0]), 0), max(min(27, y+v[1]), 0)))
                
                for i in range(28):
                    for j in range(28):
                        if grid[i][j] == 0:
                            continue
                        g = int(grid[i][j]*255)
                        pygame.draw.rect(screen, (g, g, g), pygame.Rect(30+i*10, 30+j*10, 10, 10))
                visualizeModel(model, screen, toInput(grid))

        if event.type == pygame.MOUSEBUTTONUP:
            
            pygame.display.flip()

        if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
            grid = [[0]*28 for i in range(28)]
            pygame.draw.rect(screen, background, pygame.Rect(20, 20, 300, 300))
            pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(20, 20, 300, 300), 1)
            pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(29, 29, 282, 282), 1)
            visualizeModel(model, screen, toInput(grid))





        if event.type == pygame.QUIT:
            running = False



