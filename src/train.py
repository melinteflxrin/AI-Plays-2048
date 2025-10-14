import os
import time
import pygame
from collections import deque

from game_gym import Game2048Env
from dqn_agent import DQNAgent
from game_gui import GameGUI, COLORS


class VisualTrainer(GameGUI):
    """visual training interface using GameGUI"""
    
    def __init__(self):
        # initialize pygame GUI
        super().__init__()
        
        # override window title and size for training
        pygame.display.set_caption("2048 AI Training")
        
        # extend window width to show training stats
        self.stats_width = 300
        self.window_width = 4 * self.cell_size + 5 * self.cell_margin + self.stats_width
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        
        # training statistics
        self.generation_scores = deque(maxlen=100)
        self.generation_rewards = deque(maxlen=100)
        self.current_generation = 0
        self.best_score = 0
        self.start_time = time.time()
        
        # display settings
        self.move_delay = 0.2  # seconds between moves
        self.show_training = True
        self.paused = False
        
        # training colors
        self.TRAIN_COLORS = {
            'highlight': (34, 197, 94),  # Green for AI moves
            'warning': (239, 68, 68),    # Red for game over
            'stats_bg': (240, 240, 240), # Light gray for stats panel
            'stats_border': (200, 200, 200) # Border for stats panel
        }
    
    def draw_header(self):
        """override to show training info instead of manual controls"""
        if self.show_training:
            self.draw_training_header()
        else:
            super().draw_header()
    
    def draw_training_header(self):
        """draw training-specific header"""
        y_pos = 10
        
        # title
        title_text = "AI Training Mode"
        title_surface = self.font_large.render(title_text, True, COLORS['text_dark'])
        self.screen.blit(title_surface, (20, y_pos))
        y_pos += 45
        
        # current generation and score
        generation_text = f"Generation {self.current_generation} | Score: {self.game.score}"
        generation_surface = self.font_medium.render(generation_text, True, COLORS['text_dark'])
        self.screen.blit(generation_surface, (20, y_pos))
        y_pos += 25
        
        # controls for training
        if self.paused:
            status_text = "PAUSED - Press SPACE to resume"
            color = self.TRAIN_COLORS['warning']
        else:
            status_text = "SPACE: Pause | UP/DOWN: Speed | ESC: Stop"
            color = COLORS['text_dark']
        
        controls_surface = self.font_small.render(status_text, True, color)
        self.screen.blit(controls_surface, (20, y_pos))
        y_pos += 20
        
        # speed indicator
        speed_text = f"Move delay: {self.move_delay:.2f}s"
        speed_surface = self.font_small.render(speed_text, True, COLORS['text_dark'])
        self.screen.blit(speed_surface, (20, y_pos))
    
    def draw_stats_panel(self, agent):
        """draw training statistics panel"""
        # panel position (right side of screen)
        panel_x = 4 * self.cell_size + 6 * self.cell_margin + 20
        panel_y = self.header_height + 10
        panel_width = self.stats_width - 30
        panel_height = self.window_height - self.header_height - 20
        
        # panel background
        panel_rect = pygame.Rect(panel_x - 10, panel_y - 10, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.TRAIN_COLORS['stats_bg'], panel_rect)
        pygame.draw.rect(self.screen, self.TRAIN_COLORS['stats_border'], panel_rect, 2)
        
        y_pos = panel_y + 10
        
        # title
        title_surface = self.font_medium.render("Training Stats", True, COLORS['text_dark'])
        self.screen.blit(title_surface, (panel_x, y_pos))
        y_pos += 35
        
        # best score
        best_text = f"Best Score: {self.best_score}"
        best_surface = self.font_small.render(best_text, True, self.TRAIN_COLORS['highlight'])
        self.screen.blit(best_surface, (panel_x, y_pos))
        y_pos += 25
        
        # average scores
        if self.generation_scores:
            avg_score = sum(self.generation_scores) / len(self.generation_scores)
            avg_text = f"Avg Score: {avg_score:.1f}"
            avg_surface = self.font_small.render(avg_text, True, COLORS['text_dark'])
            self.screen.blit(avg_surface, (panel_x, y_pos))
        y_pos += 25
        
        # average rewards
        if self.generation_rewards:
            avg_reward = sum(self.generation_rewards) / len(self.generation_rewards)
            reward_text = f"Avg Reward: {avg_reward:.1f}"
            reward_surface = self.font_small.render(reward_text, True, COLORS['text_dark'])
            self.screen.blit(reward_surface, (panel_x, y_pos))
        y_pos += 25
        
        # agent exploration rate
        epsilon_text = f"Exploration: {agent.epsilon:.3f}"
        epsilon_surface = self.font_small.render(epsilon_text, True, COLORS['text_dark'])
        self.screen.blit(epsilon_surface, (panel_x, y_pos))
        y_pos += 25
        
        # training time
        elapsed = time.time() - self.start_time
        time_text = f"Time: {elapsed:.0f}s"
        time_surface = self.font_small.render(time_text, True, COLORS['text_dark'])
        self.screen.blit(time_surface, (panel_x, y_pos))
        y_pos += 40
        
        # recent performance (simple text-based chart)
        if len(self.generation_scores) >= 2:
            chart_title = self.font_small.render("Recent Generations:", True, COLORS['text_dark'])
            self.screen.blit(chart_title, (panel_x, y_pos))
            y_pos += 25
            
            # show last 5 scores
            recent_scores = list(self.generation_scores)[-5:]
            for i, score in enumerate(recent_scores):
                generation_num = len(self.generation_scores) - len(recent_scores) + i + 1
                score_line = f"#{generation_num}: {score}"
                color = self.TRAIN_COLORS['highlight'] if score == self.best_score else COLORS['text_dark']
                score_surface = pygame.font.Font(None, 18).render(score_line, True, color)
                self.screen.blit(score_surface, (panel_x, y_pos))
                y_pos += 18
    
    def draw_board(self):
        """override to include stats panel"""
        # draw game board (left side)
        super().draw_board()
        
        # draw stats panel (right side) - will be called from training loop
        pass
    
    def handle_training_events(self):
        """handle events during training"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("Training paused" if self.paused else "Training resumed")
                elif event.key == pygame.K_UP:
                    # speed up (decrease delay)
                    self.move_delay = max(0.01, self.move_delay - 0.05)
                    print(f"Speed up! Delay: {self.move_delay:.2f}s")
                elif event.key == pygame.K_DOWN:
                    # slow down (increase delay)
                    self.move_delay = min(1.0, self.move_delay + 0.05)
                    print(f"Slow down! Delay: {self.move_delay:.2f}s")
        return True
    
    def visual_train(self, generations=500, save_frequency=100):
        """Visual training with real-time display"""
        print("Starting Visual 2048 Training!")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  UP/DOWN: Adjust speed")
        print("  ESC: Stop training")
        print("=" * 50)
        
        # create environment and agent
        env = Game2048Env()
        agent = DQNAgent()
        
        # ensure models directory exists
        os.makedirs("../models", exist_ok=True)

        # training state
        self.current_generation = 0
        self.best_score = 0
        self.start_time = time.time()
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        running = True
        
        for generation in range(1, generations + 1):
            if not running:
                break
                
            self.current_generation = generation
            
            # reset environment and sync with GUI game
            observation, info = env.reset()
            self.game = env.game  # sync the GUI game with environment game
            
            total_reward = 0
            steps = 0
            
            while running:
                # handle events
                if not self.handle_training_events():
                    running = False
                    break
                
                # pause handling
                while self.paused and running:
                    if not self.handle_training_events():
                        running = False
                        break
                    
                    # draw paused state
                    self.draw_board()
                    self.draw_stats_panel(agent)
                    pygame.display.flip()
                    self.clock.tick(30)
                
                if not running:
                    break
                
                # agent chooses action
                action = agent.choose_action(observation)
                
                # take action in environment
                next_observation, reward, terminated, truncated, info = env.step(action)
                
                # store experience and learn
                agent.remember(observation, action, reward, next_observation, terminated)
                agent.learn()

                # update state
                observation = next_observation
                total_reward += reward
                steps += 1
                
                # draw current state
                self.draw_board()
                self.draw_stats_panel(agent)
                
                # show current action
                action_text = f"AI Action: {action_names[action]}"
                action_surface = self.font_small.render(action_text, True, self.TRAIN_COLORS['highlight'])
                # stats panel area on the right
                stats_panel_x = 4 * self.cell_size + 6 * self.cell_margin + 20
                self.screen.blit(action_surface, (stats_panel_x, self.header_height - 25))
                
                pygame.display.flip()
                
                # control training speed
                if self.move_delay > 0:
                    pygame.time.wait(int(self.move_delay * 1000))
                
                # episode ends
                if terminated or truncated:
                    break
            
            if not running:
                break
            
            # generation completed - update statistics
            final_score = info['score']
            self.generation_scores.append(final_score)
            self.generation_rewards.append(total_reward)
            
            # save best model
            print(f"Generation {generation}: Score = {final_score}, Best = {self.best_score}")
            if final_score > self.best_score:
                self.best_score = final_score
                model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "dqn_2048_best.pth")
                agent.save_model(model_path)
                print(f"*** New best score: {self.best_score} (Generation {generation}) ***")
            
            # decay epsilon once per generation
            agent.decay_epsilon()
            
            # print progress
            if generation % save_frequency == 0:
                avg_score = sum(list(self.generation_scores)[-100:]) / min(len(self.generation_scores), 100)
                print(f"Generation {generation:4d} | Score: {final_score:4.0f} | "
                      f"Avg: {avg_score:6.1f} | Epsilon: {agent.epsilon:.3f}")
        
        # save last model and cleanup
        if running:
            agent.save_model("../models/dqn_2048_last.pth")
            print(f"\nTraining completed! Best score: {self.best_score}")
        else:
            # save model even when stopped early
            agent.save_model("../models/dqn_2048_last.pth")
            print(f"\nTraining stopped early. Best score: {self.best_score}")
            print("Model saved to ../models/dqn_2048_last.pth")
        
        pygame.quit()
        return agent, list(self.generation_scores), list(self.generation_rewards)


if __name__ == "__main__":
    generations = 500
    
    try:
        visual_trainer = VisualTrainer()
        visual_trainer.visual_train(generations=generations)
        print("\nTraining completed!")
        
    except Exception as e:
        print(f"Error during visual training: {e}")
        print("Make sure all dependencies are installed: pip install -r ../requirements.txt")