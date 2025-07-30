import pygame
import sys
from game import Game2048


pygame.init()


COLORS = {
    'background': (248, 250, 252),        
    'grid_background': (165, 174, 185),   
    'empty_cell': (203, 213, 225),        
    'text_dark': (51, 65, 85),            
    'text_light': (255, 255, 255),        
    # tile colors
    2: (219, 234, 254),                   
    4: (191, 219, 254),                   
    8: (147, 197, 253),                   
    16: (96, 165, 250),                   
    32: (59, 130, 246),                   
    64: (37, 99, 235),                   
    128: (29, 78, 216),                   
    256: (30, 64, 175),                  
    512: (30, 58, 138),                  
    1024: (23, 37, 84),                   
    2048: (15, 23, 42),                   
}


class GameGUI:
    def __init__(self):
        """initialize game GUI"""
        self.game = Game2048()
        
        # GUI settings
        self.cell_size = 100
        self.cell_margin = 10
        self.header_height = 120
        
        # window size
        grid_size = 4 * self.cell_size + 5 * self.cell_margin
        self.window_width = grid_size
        self.window_height = grid_size + self.header_height
        
        # create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("2048 Game")
        
        # fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # game clock
        self.clock = pygame.time.Clock()
    
    def get_tile_color(self, value):
        """get background color for a tile value"""
        if value in COLORS:
            return COLORS[value]
        elif value > 2048:
            return COLORS[2048]  # 2048 color for higher values
        else:
            return COLORS['empty_cell']
    
    def get_text_color(self, value):
        """get text color for a tile value"""
        if value <= 4:
            return COLORS['text_dark']
        else:
            return COLORS['text_light']
    
    def draw_board(self):
        """draw the game board"""
        # clear screen with background color
        self.screen.fill(COLORS['background'])
        
        self.draw_header()
        
        # draw the grid background
        grid_y = self.header_height
        grid_rect = pygame.Rect(0, grid_y, self.window_width, self.window_width)
        pygame.draw.rect(self.screen, COLORS['grid_background'], grid_rect)
        
        # draw cells
        for row in range(4):
            for col in range(4):
                self.draw_cell(row, col)
    
    def draw_header(self):
        """draw the header with score and instructions"""
        # draw score
        score_text = self.font_large.render(f"Score: {self.game.score}", True, COLORS['text_dark'])
        self.screen.blit(score_text, (20, 20))
        
        # instructions
        if self.game.game_over:
            instruction_text = "Game Over! Press R to restart"
            color = (200, 0, 0)  # Red
        else:
            instruction_text = "Use arrow keys to move tiles"
            color = COLORS['text_dark']
        
        instruction_surface = self.font_small.render(instruction_text, True, color)
        self.screen.blit(instruction_surface, (20, 70))
        
        # restart instruction
        restart_text = self.font_small.render("Press R to restart, ESC to quit", True, COLORS['text_dark'])
        self.screen.blit(restart_text, (20, 95))
    
    def draw_cell(self, row, col):
        """draw a single cell of the grid"""
        value = self.game.board[row][col]
        
        # cell position
        x = col * (self.cell_size + self.cell_margin) + self.cell_margin
        y = row * (self.cell_size + self.cell_margin) + self.cell_margin + self.header_height
        
        # draw cell background
        cell_rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
        cell_color = self.get_tile_color(value)
        pygame.draw.rect(self.screen, cell_color, cell_rect, border_radius=8)
        
        # draw cell value (if not empty)
        if value != 0:
            text_color = self.get_text_color(value)
            
            # choose font size based on number of digits
            if value < 100:
                font = self.font_large
            elif value < 1000:
                font = self.font_medium
            else:
                font = self.font_small
            
            text_surface = font.render(str(value), True, text_color)
            
            # center the text in the cell
            text_rect = text_surface.get_rect()
            text_rect.center = (x + self.cell_size // 2, y + self.cell_size // 2)
            self.screen.blit(text_surface, text_rect)
    
    def handle_keypress(self, key):
        """keyboard input"""
        if key == pygame.K_ESCAPE:
            return False  # quit
        
        elif key == pygame.K_r:
            self.game.reset()
            print("Game restarted!")
        
        elif not self.game.game_over:
            move_made = False
            points = 0
            
            if key == pygame.K_LEFT:
                move_made, points = self.game.make_move('left')
            elif key == pygame.K_RIGHT:
                move_made, points = self.game.make_move('right')
            elif key == pygame.K_UP:
                move_made, points = self.game.make_move('up')
            elif key == pygame.K_DOWN:
                move_made, points = self.game.make_move('down')
            
        return True  # continue
    
    def run(self):
        """main loop"""
        print("2048 Game Started!")
        print("Use arrow keys to move tiles")
        print("Press R to restart, ESC to quit")
        print()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_keypress(event.key)
            
            self.draw_board()
            
            # update display
            pygame.display.flip()
            
            # frame rate
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()


def main():
    try:
        game = GameGUI()
        game.run()
    except Exception as e:
        print(f"Error running game: {e}")
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
