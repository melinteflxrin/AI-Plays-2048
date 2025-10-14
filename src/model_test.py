import pygame
import os

from game_gym import Game2048Env
from dqn_agent import DQNAgent
from game_gui import GameGUI, COLORS


class ModelTester(GameGUI):
    """watch trained models play"""
    
    def __init__(self):
        """initialize AI viewer"""
        super().__init__()
        
        # setup
        pygame.display.set_caption("2048 AI Model Test")
        self.move_delay = 0.5  # seconds between moves
        
        # load AI agent
        self.agent = DQNAgent()
        self.agent.load_model("../models/dqn_2048_best.pth")
        self.agent.epsilon = 0.0  # no exploration
        
        # colors for AI display
        self.AI_COLORS = {
            'highlight': (34, 197, 94),  # green for AI moves
            'warning': (239, 68, 68),    # red for game over
        }
    
    def draw_header(self):
        """show AI game info"""
        y_pos = 20
        
        # title and score
        title_text = f"AI Playing 2048 - Score: {self.game.score}"
        title_surface = self.font_large.render(title_text, True, COLORS['text_dark'])
        self.screen.blit(title_surface, (20, y_pos))
        y_pos += 45
        
        # controls
        controls_text = "SPACE: Pause | UP/DOWN: Speed | ESC: Quit"
        controls_surface = self.font_small.render(controls_text, True, COLORS['text_dark'])
        self.screen.blit(controls_surface, (20, y_pos))
    
    def handle_events(self):
        """handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_UP:
                    # speed up
                    self.move_delay = max(0.1, self.move_delay - 0.1)
                    print(f"Faster! Delay: {self.move_delay:.1f}s")
                elif event.key == pygame.K_DOWN:
                    # slow down
                    self.move_delay = min(2.0, self.move_delay + 0.1)
                    print(f"Slower! Delay: {self.move_delay:.1f}s")
        return True
    
    def play_single_game(self):
        """watch AI play one game"""
        print("\nStarting new game...")
        
        # create environment and reset
        env = Game2048Env()
        observation, info = env.reset()
        self.game = env.game  # sync GUI with environment
        
        paused = False
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        while True:
            # handle user input
            if not self.handle_events():
                return "quit"
            
            # check for pause
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                paused = not paused
                print("Paused" if paused else "Resumed")
                pygame.time.wait(200)  # prevent rapid toggling
            
            # pause handling
            if paused:
                self.draw_board()
                pause_text = "PAUSED"
                pause_surface = self.font_large.render(pause_text, True, self.AI_COLORS['warning'])
                self.screen.blit(pause_surface, (20, self.header_height + 50))
                pygame.display.flip()
                self.clock.tick(30)
                continue
            
            # AI makes move
            action = self.agent.choose_action(observation)
            action_name = action_names[action]
            
            # show current state with AI action
            self.draw_board()
            action_text = f"AI Move: {action_name}"
            action_surface = self.font_medium.render(action_text, True, self.AI_COLORS['highlight'])
            self.screen.blit(action_surface, (20, self.header_height + 20))
            pygame.display.flip()
            
            # wait for move delay
            pygame.time.wait(int(self.move_delay * 1000))
            
            # execute move
            observation, reward, terminated, truncated, info = env.step(action)
            
            # update display
            self.draw_board()
            pygame.display.flip()
            
            # check if game over
            if terminated or truncated:
                final_score = info['score']
                
                # show game over message
                game_over_text = f"GAME OVER - Final Score: {final_score}"
                game_over_surface = self.font_large.render(game_over_text, True, self.AI_COLORS['warning'])
                self.screen.blit(game_over_surface, (20, self.header_height + 80))
                pygame.display.flip()
                
                # wait to show final state
                pygame.time.wait(3000)
                
                return final_score


def main():
    print("2048 AI Model Test!")
    print("Controls: SPACE=Pause | UP/DOWN=Speed | ESC=Quit")
    
    # check for best model
    best_model_path = "../models/dqn_2048_best.pth"
    
    if not os.path.exists(best_model_path):
        print("No best model found!")
        print("Train a model first using: python train.py")
        return
    
    try:
        tester = ModelTester()
        
        while True:
            # play one game
            final_score = tester.play_single_game()
            
            if final_score == "quit":
                break
                
            # ask for replay
            print(f"\nFinal Score: {final_score}")
            replay = input("Play again? (y/n): ").strip().lower()
            
            if replay != 'y':
                break
        
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()