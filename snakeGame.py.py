import pygame 
import time 
import random
import pygame.image as img

pygame.init()


white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
orange = (255, 165, 0)

width , height = 600, 400

game_display = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game")

clock = pygame.time.Clock()

snake_size = 10
snake_speed = 5

message_font = pygame.font.SysFont('ubuntu', 30)
score_font = pygame.font.SysFont('ubuntu', 25)

def print_score(score):
    text = score_font.render("Score: " + str(score), True, orange)
    game_display.blit(text, [0,0])

def draw_snake(snake_size, snake_pixels):
    for pixel in snake_pixels:
        pygame.draw.rect(game_display, white, [pixel[0], pixel[1], snake_size, snake_size])

# def render_game_state():
#     game_state_surface = pygame.Surface((width, height))
    
#     game_state_surface.fill(black)
#     pygame.draw.rect(game_state_surface, orange, [target_x, target_y, snake_size, snake_size])
#     draw_snake(snake_size, snake_pixels)
    
#     return game_state_surface
# def capture_screen_and_save():
#     game_state_surface = render_game_state()

#     img.save(game_state_surface, 'game_state.png')    

def run_game():
    game_over = False
    game_close = False 

    x = width/2
    y = height/2

    x_speed = 0
    y_speed = 0

    snake_pixels = []
    snake_length = 1
    target_x  = round(random.randrange(0, width-snake_size)/10.0) * 10.0
    target_y  = round(random.randrange(0, height-snake_size)/10.0) * 10.0

    counter = 0
    current_direction = "RIGHT"
    while not game_over:

        while game_close:
            game_display.fill(black)
            game_over_message = message_font.render("Game Over!", True, red)
            game_display.blit(game_over_message, [width/3, height/3])
            print_score(snake_length - 1)
            pygame.display.update()

            for event in pygame.event.get():               
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        game_over = True
                        game_close = False
                  
                    if event.key == pygame.K_2:
                        run_game()

                if event.key == pygame.QUIT:
                        game_over = True
                        game_close = False    

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_speed = -snake_size
                    y_speed = 0

                if event.key == pygame.K_RIGHT:
                    x_speed = snake_size
                    y_speed = 0

                if event.key == pygame.K_UP:
                    x_speed = 0
                    y_speed = -snake_size    

                if event.key == pygame.K_DOWN:
                    x_speed = 0
                    y_speed = snake_size   

        if x >= width or x < 0 or y >= height or y < 0:
            game_close = True

        x += x_speed
        y += y_speed

        game_display.fill(black)
        pygame.draw.rect(game_display, orange, [target_x, target_y, snake_size, snake_size])

        snake_pixels.append([x,y])

        if len(snake_pixels) > snake_length:
            del snake_pixels[0]

        for pixel in snake_pixels[:-1]:
            if pixel == [x,y]:
                game_close = True

        draw_snake(snake_size, snake_pixels)
        print_score(snake_length - 1)

        # counter += 1  
        # if counter >= 5:
        #     if current_direction == "RIGHT":
        #         x_speed = snake_size
        #         y_speed = 0
        #         current_direction = "DOWN" 
        #     elif current_direction == "DOWN":
        #         x_speed = 0
        #         y_speed = snake_size
        #         current_direction = "RIGHT"  
        #     counter = 0  

        pygame.display.update()

        # counter += 1  
        # if counter >= 10:
        #     x_speed = snake_size
        #     y_speed = 0
        #     counter = 0  


        if x == target_x and y == target_y:
            target_x  = round(random.randrange(0, width-snake_size)/10.0) * 10.0
            target_y  = round(random.randrange(0, height-snake_size)/10.0) * 10.0

            snake_length += 1

        clock.tick(snake_speed)    

    pygame.quit()
    quit()

run_game()   






