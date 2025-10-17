"""
Model Test Script - Watch your trained TD agent play 2048
Uses your existing GameGUI implementation
"""
import pygame
import os
import numpy as np

from game_gym import Game2048Env
from td_agent import TDAgent
from game_gui import GameGUI, COLORS


class ModelViewer(GameGUI):
    """Extends GameGUI to watch trained models play"""
    
    def __init__(self, model_path):
        """Initialize viewer with trained model"""
        super().__init__()
        
        # Load agent
        print(f"Loading model from: {model_path}")
        self.agent = TDAgent(
            tuple_patterns='8x6',
            learning_method='OTD',
            v_init=0,
            learning_rate=0.1
        )
        self.agent.load_model(model_path)
        print("Model loaded successfully!\n")
        
        pygame.display.set_caption("2048 AI - Model Viewer")
        
        # Viewing settings
        self.move_delay = 300  # milliseconds between moves
        self.paused = False
        
    def draw_info(self, move_count, max_tile, action_name=""):
        """Draw additional info for AI viewer"""
        # Draw score and game info (reuse parent's draw_header)
        self.draw_header()
        
        # Add AI-specific info
        y_pos = 60
        
        # Max tile and moves
        info_text = f"Max Tile: {max_tile} | Moves: {move_count}"
        info_surface = self.font_medium.render(info_text, True, COLORS['text_dark'])
        self.screen.blit(info_surface, (20, y_pos))
        y_pos += 40
        
        # Current action or status
        if self.paused:
            status_text = "PAUSED - Press SPACE to continue"
            color = (239, 68, 68)  # Red
        elif action_name:
            status_text = f"AI Move: {action_name}"
            color = (34, 197, 94)  # Green
        else:
            status_text = "Controls: SPACE=Pause | UP/DOWN=Speed | R=Restart | ESC=Quit"
            color = COLORS['text_dark']
        
        status_surface = self.font_small.render(status_text, True, color)
        self.screen.blit(status_surface, (20, y_pos))
    
    def play_game(self):
        """Watch AI play one complete game"""
        env = Game2048Env()
        observation, info = env.reset()
        self.game = env.game  # Sync GUI with environment
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        max_tile = 0
        move_count = 0
        current_action = ""
        
        print("Game started! Watching AI play...")
        
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "quit"
                    elif event.key == pygame.K_r:
                        return "restart"
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        print("Paused" if self.paused else "Resumed")
                    elif event.key == pygame.K_UP:
                        self.move_delay = max(50, self.move_delay - 50)
                        print(f"Faster! Delay: {self.move_delay}ms")
                    elif event.key == pygame.K_DOWN:
                        self.move_delay = min(2000, self.move_delay + 100)
                        print(f"Slower! Delay: {self.move_delay}ms")
            
            # Draw current state
            self.draw_board()
            self.draw_info(move_count, max_tile, current_action)
            pygame.display.flip()
            
            if self.paused:
                self.clock.tick(30)
                continue
            
            # AI makes decision
            action = self.agent.choose_action(observation)
            current_action = action_names[action]
            move_count += 1
            
            # Show decision briefly
            self.draw_board()
            self.draw_info(move_count, max_tile, current_action)
            pygame.display.flip()
            pygame.time.wait(self.move_delay)
            
            # Execute move
            observation, reward, terminated, truncated, info = env.step(action)
            self.game = env.game  # Keep GUI synced
            
            # Update max tile
            current_max = int(np.max(observation))
            if current_max > max_tile:
                max_tile = current_max
                print(f"New max tile: {max_tile}")
            
            # Check if game over
            if terminated or truncated:
                final_score = env.game.score
                
                # Draw final state
                self.draw_board()
                self.draw_info(move_count, max_tile, "")
                
                # Game over message
                game_over_text = "GAME OVER"
                game_over_surface = self.font_large.render(game_over_text, True, (239, 68, 68))
                game_over_rect = game_over_surface.get_rect(center=(self.window_width // 2, 
                                                                     self.header_height + 200))
                self.screen.blit(game_over_surface, game_over_rect)
                pygame.display.flip()
                
                print(f"\nGame Over!")
                print(f"Final Score: {final_score:,}")
                print(f"Max Tile: {max_tile}")
                print(f"Moves: {move_count}\n")
                
                # Wait before closing
                pygame.time.wait(3000)
                
                return {
                    "score": final_score,
                    "max_tile": max_tile,
                    "moves": move_count
                }
            
            self.clock.tick(60)


def main():
    """Main function to run model viewer"""
    print("\n" + "="*70)
    print("2048 AI Model Viewer")
    print("="*70)
    print("\nControls:")
    print("  SPACE: Pause/Resume")
    print("  UP/DOWN: Adjust speed")
    print("  R: Restart game")
    print("  ESC: Quit")
    print("="*70 + "\n")
    
    # Find model file
    model_options = [
        "../models/td_2048_ep24000_8x6.pkl",
        "../models/td_2048_best_8x6.pkl",
        "models/td_2048_ep24000_8x6.pkl",
        "models/td_2048_best_8x6.pkl"
    ]
    
    model_path = None
    for path in model_options:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("Error: No trained model found!")
        print("Expected model at: ../models/td_2048_ep24000_8x6.pkl")
        print("Train a model first using: python train.py")
        return
    
    try:
        viewer = ModelViewer(model_path)
        
        # Track statistics across games
        games_played = 0
        total_score = 0
        max_tiles_achieved = {}
        
        while True:
            result = viewer.play_game()
            
            if result == "quit":
                break
            elif result == "restart":
                continue
            
            # Update statistics
            games_played += 1
            total_score += result['score']
            max_tile = result['max_tile']
            max_tiles_achieved[max_tile] = max_tiles_achieved.get(max_tile, 0) + 1
            
            # Show statistics
            print("="*50)
            print(f"Games Played: {games_played}")
            print(f"Average Score: {total_score / games_played:,.0f}")
            print(f"\nMax Tiles Achieved:")
            for tile in sorted(max_tiles_achieved.keys(), reverse=True):
                count = max_tiles_achieved[tile]
                percentage = (count / games_played) * 100
                print(f"  {tile:5d}: {count:3d} times ({percentage:5.1f}%)")
            print("="*50 + "\n")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        print("\nThanks for watching!")


if __name__ == "__main__":
    main()
