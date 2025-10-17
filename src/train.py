import os
import time
import pygame
import numpy as np
import copy
from collections import deque

from game_gym import Game2048Env
from td_agent import TDAgent  # Using TD agent instead of DQN
from game_gui import GameGUI, COLORS


class VisualTrainer(GameGUI):
    """visual training interface using GameGUI"""
    
    def __init__(self):
        # initialize pygame GUI
        super().__init__()
        
        # override window title and size for training
        pygame.display.set_caption("2048 AI Training - TD Learning with N-Tuple Network")
        
        # extend window width to show training stats
        self.stats_width = 350  # Increased for more stats
        self.window_width = 4 * self.cell_size + 5 * self.cell_margin + self.stats_width
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        
        # training statistics
        self.generation_scores = deque(maxlen=100)
        self.generation_rewards = deque(maxlen=100)
        self.generation_max_tiles = deque(maxlen=100)  # Track max tiles reached
        self.current_generation = 0
        self.best_score = 0
        self.best_max_tile = 0
        self.start_time = time.time()
        
        # display settings
        self.move_delay = 0.1  # seconds between moves (faster for TD learning)
        self.show_training = True
        self.paused = False
        
        # training colors
        self.TRAIN_COLORS = {
            'highlight': (34, 197, 94),  # Green for AI moves
            'warning': (239, 68, 68),    # Red for game over
            'stats_bg': (240, 240, 240), # Light gray for stats panel
            'stats_border': (200, 200, 200), # Border for stats panel
            'exploration': (59, 130, 246), # Blue for exploration phase
            'finetune': (168, 85, 247)    # Purple for fine-tuning phase
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
        title_text = "TD Learning - N-Tuple Network"
        title_surface = self.font_large.render(title_text, True, COLORS['text_dark'])
        self.screen.blit(title_surface, (20, y_pos))
        y_pos += 45
        
        # current generation and score
        generation_text = f"Episode {self.current_generation} | Score: {self.game.score}"
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
        
        # Learning method and phase
        method_text = f"Method: {agent.learning_method}"
        method_surface = self.font_small.render(method_text, True, COLORS['text_dark'])
        self.screen.blit(method_surface, (panel_x, y_pos))
        y_pos += 20
        
        phase_color = self.TRAIN_COLORS['exploration'] if agent.phase == 'exploration' else self.TRAIN_COLORS['finetune']
        phase_text = f"Phase: {agent.phase.upper()}"
        phase_surface = self.font_small.render(phase_text, True, phase_color)
        self.screen.blit(phase_surface, (panel_x, y_pos))
        y_pos += 25
        
        # best score
        best_text = f"Best Score: {self.best_score}"
        best_surface = self.font_small.render(best_text, True, self.TRAIN_COLORS['highlight'])
        self.screen.blit(best_surface, (panel_x, y_pos))
        y_pos += 20
        
        # best max tile
        if self.best_max_tile > 0:
            tile_text = f"Best Tile: {self.best_max_tile}"
            tile_surface = self.font_small.render(tile_text, True, self.TRAIN_COLORS['highlight'])
            self.screen.blit(tile_surface, (panel_x, y_pos))
        y_pos += 25
        
        # average scores
        if self.generation_scores:
            avg_score = sum(self.generation_scores) / len(self.generation_scores)
            avg_text = f"Avg Score: {avg_score:.1f}"
            avg_surface = self.font_small.render(avg_text, True, COLORS['text_dark'])
            self.screen.blit(avg_surface, (panel_x, y_pos))
        y_pos += 20
        
        # average max tile
        if self.generation_max_tiles:
            avg_tile = sum(self.generation_max_tiles) / len(self.generation_max_tiles)
            tile_text = f"Avg Max Tile: {avg_tile:.0f}"
            tile_surface = self.font_small.render(tile_text, True, COLORS['text_dark'])
            self.screen.blit(tile_surface, (panel_x, y_pos))
        y_pos += 25
        
        # Learning rate
        lr_text = f"Learning Rate: {agent.learning_rate:.4f}"
        lr_surface = self.font_small.render(lr_text, True, COLORS['text_dark'])
        self.screen.blit(lr_surface, (panel_x, y_pos))
        y_pos += 25
        
        # Network statistics
        net_stats = agent.get_network_statistics()
        features_text = f"Features: {net_stats['total_features']:,}"
        features_surface = pygame.font.Font(None, 18).render(features_text, True, COLORS['text_dark'])
        self.screen.blit(features_surface, (panel_x, y_pos))
        y_pos += 18
        
        # Exploration statistics
        exp_stats = agent.get_exploration_stats()
        if exp_stats.get('total_updates', 0) > 0:
            pos_ratio = exp_stats['positive_ratio'] * 100
            exp_text = f"TD+ ratio: {pos_ratio:.1f}%"
            exp_surface = pygame.font.Font(None, 18).render(exp_text, True, COLORS['text_dark'])
            self.screen.blit(exp_surface, (panel_x, y_pos))
        y_pos += 20
        
        # training time
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        time_text = f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}"
        time_surface = self.font_small.render(time_text, True, COLORS['text_dark'])
        self.screen.blit(time_surface, (panel_x, y_pos))
        y_pos += 40
        
        # recent performance (simple text-based chart)
        if len(self.generation_scores) >= 2:
            chart_title = self.font_small.render("Recent Episodes:", True, COLORS['text_dark'])
            self.screen.blit(chart_title, (panel_x, y_pos))
            y_pos += 25
            
            # show last 5 scores with max tiles
            recent_count = min(5, len(self.generation_scores))
            recent_scores = list(self.generation_scores)[-recent_count:]
            recent_tiles = list(self.generation_max_tiles)[-recent_count:]
            
            for i, (score, tile) in enumerate(zip(recent_scores, recent_tiles)):
                generation_num = len(self.generation_scores) - recent_count + i + 1
                score_line = f"#{generation_num}: {score} [{int(tile)}]"
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
    
    def visual_train(self, 
                    episodes=500, 
                    save_frequency=100,
                    tuple_patterns='4x6',
                    learning_method='OTD+TC',
                    v_init=320000):
        """
        Visual training with real-time display using TD learning
        
        Args:
            episodes: Number of training episodes
            save_frequency: Save model every N episodes
            tuple_patterns: '4x6' (faster, ~280k score) or '8x6' (better, ~371k score)
            learning_method: 'OTD', 'OTC', or 'OTD+TC' (hybrid, recommended)
            v_init: Optimistic initialization value (320000 works well)
        """
        print("\n" + "="*70)
        print("Starting Visual 2048 Training - TD Learning with N-Tuple Network")
        print("="*70)
        print(f"Configuration:")
        print(f"  - Network: {tuple_patterns}-tuple")
        print(f"  - Method: {learning_method}")
        print(f"  - Episodes: {episodes}")
        print(f"  - Optimistic Init: {v_init:,}")
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  UP/DOWN: Adjust speed")
        print("  ESC: Stop training")
        print("="*70 + "\n")
        
        # create environment and agent
        env = Game2048Env()
        
        # Set learning rate based on method
        if learning_method == 'OTC':
            learning_rate = 1.0  # TC learning uses higher α
        else:
            learning_rate = 0.1  # OTD uses α=0.1
        
        agent = TDAgent(
            tuple_patterns=tuple_patterns,
            learning_method=learning_method,
            v_init=v_init,
            learning_rate=learning_rate,
            fine_tune_ratio=0.1  # Last 10% for fine-tuning in OTD+TC
        )
        
        # Set total episodes for learning rate scheduling
        agent.set_total_training_episodes(episodes)
        
        # ensure models directory exists
        os.makedirs("../models", exist_ok=True)

        # training state
        self.current_generation = 0
        self.best_score = 0
        self.best_max_tile = 0
        self.start_time = time.time()
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        running = True
        
        for episode in range(1, episodes + 1):
            if not running:
                break
                
            self.current_generation = episode
            
            # reset environment and sync with GUI game
            observation, info = env.reset()
            self.game = env.game  # sync the GUI game with environment game
            
            episode_history = []  # Store (afterstate, reward, next_state, done) for learning
            episode_score = 0
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
                
                # agent chooses action using TD policy
                board = copy.deepcopy(env.game.board)
                action = agent.choose_action(board)
                
                # Get afterstate before taking action
                afterstate_board, move_reward, valid = env.get_afterstate(action)
                
                # take action in environment
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_board = env.game.board
                
                # Store for learning if move was valid
                if valid:
                    episode_history.append((afterstate_board, move_reward, next_board, terminated))
                
                # update state
                observation = next_observation
                episode_score = info['score']
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
            
            # After episode ends, perform TD learning update on the history we just collected
            agent._backward_update(episode_history)
            agent.episodes_trained += 1
            agent._update_learning_rate()
            agent._update_phase()
            
            # Update statistics
            final_score = episode_score
            max_tile = int(np.max(env.game.board))
            
            self.generation_scores.append(final_score)
            self.generation_rewards.append(final_score)  # For compatibility
            self.generation_max_tiles.append(max_tile)
            
            # Track best results
            if final_score > self.best_score:
                self.best_score = final_score
                model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         "models", f"td_2048_best_{tuple_patterns}.pkl")
                agent.save_model(model_path)
                print(f"*** New best score: {self.best_score} (Episode {episode}, Max Tile: {max_tile}) ***")
            
            if max_tile > self.best_max_tile:
                self.best_max_tile = max_tile
            
            # print progress
            if episode % save_frequency == 0:
                avg_score = sum(list(self.generation_scores)[-100:]) / min(len(self.generation_scores), 100)
                avg_tile = sum(list(self.generation_max_tiles)[-100:]) / min(len(self.generation_max_tiles), 100)
                
                # Count tile achievements
                tile_counts = {}
                for tile in self.generation_max_tiles:
                    tile_counts[tile] = tile_counts.get(tile, 0) + 1
                
                print(f"\n{'='*70}")
                print(f"Episode {episode:4d} | Score: {final_score:6.0f} | Max Tile: {max_tile:5d}")
                print(f"Avg Score: {avg_score:6.1f} | Avg Max Tile: {avg_tile:6.0f}")
                print(f"Best Score: {self.best_score:6.0f} | Best Tile: {self.best_max_tile:5d}")
                print(f"Phase: {agent.phase} | LR: {agent.learning_rate:.4f}")
                
                # Show exploration stats
                exp_stats = agent.get_exploration_stats()
                print(f"TD Errors: {exp_stats['positive_ratio']*100:.1f}% positive")
                print(f"{'='*70}")
                
                # Save periodic checkpoint
                checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                              "models", f"td_2048_checkpoint_{tuple_patterns}.pkl")
                agent.save_model(checkpoint_path)
        
        # save last model and cleanup
        if running:
            final_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "models", f"td_2048_final_{tuple_patterns}.pkl")
            agent.save_model(final_path)
            print(f"\n{'='*70}")
            print(f"Training completed! Best score: {self.best_score}")
            print(f"Best max tile: {self.best_max_tile}")
            print(f"{'='*70}\n")
        else:
            # save model even when stopped early
            early_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "models", f"td_2048_stopped_{tuple_patterns}.pkl")
            agent.save_model(early_path)
            print(f"\n{'='*70}")
            print(f"Training stopped early. Best score: {self.best_score}")
            print(f"Model saved to {early_path}")
            print(f"{'='*70}\n")
        
        pygame.quit()
        return agent, list(self.generation_scores), list(self.generation_rewards)


if __name__ == "__main__":
    # Training configuration - RESUME FROM BEST MODEL with improvements
    EPISODES = 50000  # Continue training for 50k more episodes with better settings
    TUPLE_PATTERNS = '8x6'  # 8x6-tuple network (matches your best model)
    
    # LEARNING METHOD: Choose based on your needs
    # 'OTD' = Optimistic TD (stable, no fine-tuning crash risk)
    # 'OTC' = Optimistic TC (faster convergence, needs higher V_INIT)
    # 'OTD+TC' = Hybrid (best but risky with low V_INIT)
    LEARNING_METHOD = 'OTD'  # Stable learning without catastrophic forgetting
    
    # INITIALIZATION: Not used when loading existing model
    V_INIT = 0  # Only used if starting fresh (ignored when loading)
    
    # RESUME TRAINING SETTINGS
    LOAD_MODEL = True  # Set to True to resume from best model
    LOAD_MODEL_PATH = "models/td_2048_best_8x6.pkl"  # Your peak performance model
    USE_LR_DECAY = True  # CRITICAL: Prevents catastrophic interference
    INITIAL_LR = 0.025  # Start with lower LR for fine-tuning (was 0.1)
    MIN_LR = 0.005  # Don't go below this
    LR_DECAY_RATE = 0.9995  # Gradual decay: 0.025 → 0.005 over 50k episodes
    
    VISUAL_MODE = False  # Set to False for faster training without visualization
    
    print("\n" + "="*70)
    print("2048 AI Training - Research-Based TD Learning")
    print("="*70)
    print("\nBased on state-of-the-art research:")
    print("  'On Reinforcement Learning for the Game of 2048'")
    print("  by Hung Guei (PhD Dissertation, 2023)")
    print("\nThis implementation uses:")
    print("  * N-Tuple Networks (superior to DQN for 2048)")
    print("  * TD Learning with Afterstate Framework")
    print("  * Optimistic Initialization for exploration")
    print("  * Temporal Coherence for adaptive learning rates")
    print("\nExpected Performance:")
    if TUPLE_PATTERNS == '4x6':
        print(f"  - {TUPLE_PATTERNS}-tuple + OTC: ~280k avg score, 5% reach 32768-tile")
    else:
        print(f"  - {TUPLE_PATTERNS}-tuple + OTD+TC: ~371k avg score, 22% reach 32768-tile")
    print("\nNOTE: Learning takes time! The agent will:")
    print("  - Start random (exploring with optimistic values)")
    print("  - Gradually learn patterns (after ~100-1000 episodes)")
    print("  - Develop corner strategy naturally through TD learning")
    print("="*70 + "\n")
    
    try:
        if VISUAL_MODE:
            visual_trainer = VisualTrainer()
            visual_trainer.visual_train(
                episodes=EPISODES,
                save_frequency=50,
                tuple_patterns=TUPLE_PATTERNS,
                learning_method=LEARNING_METHOD,
                v_init=V_INIT
            )
        else:
            # Fast training without visualization
            if LOAD_MODEL:
                print("="*70)
                print("RESUMING TRAINING FROM BEST MODEL")
                print("="*70)
                print(f"Loading model from: {LOAD_MODEL_PATH}")
                print(f"Training for {EPISODES:,} more episodes")
                print(f"Learning rate decay: {INITIAL_LR} -> {MIN_LR}")
                print("="*70 + "\n")
            else:
                print("Running fast training mode (no visualization)...")
                print("This will be much faster for convergence.\n")
            
            env = Game2048Env()
            
            # Set initial learning rate
            if USE_LR_DECAY:
                learning_rate = INITIAL_LR
            elif LEARNING_METHOD == 'OTC':
                learning_rate = 1.0
            else:
                learning_rate = 0.1
            
            agent = TDAgent(
                tuple_patterns=TUPLE_PATTERNS,
                learning_method=LEARNING_METHOD,
                v_init=V_INIT,
                learning_rate=learning_rate,
                fine_tune_ratio=0.1
            )
            
            # Load existing model if specified
            if LOAD_MODEL:
                model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), LOAD_MODEL_PATH)
                if os.path.exists(model_path):
                    agent.load_model(model_path)
                    print(f"[OK] Model loaded successfully!\n")
                else:
                    print(f"[WARNING] Model file not found at {model_path}")
                    print("Starting fresh training instead.\n")
            
            agent.set_total_training_episodes(EPISODES)
            
            os.makedirs("../models", exist_ok=True)
            
            # Initialize best scores - higher if resuming from saved model
            if LOAD_MODEL:
                # Set high initial values so we only save when truly improving
                best_score = 45000  # Approximate score from loaded model
                best_max_tile = 4096  # We know the loaded model reached 4096
                print(f"Starting with best score threshold: {best_score:,} (from loaded model)")
                print(f"Will only save new models if they beat this score\n")
            else:
                best_score = 0
                best_max_tile = 0
            
            # Track best checkpoint performance to detect decline
            best_checkpoint_2048_rate = 0
            checkpoint_decline_count = 0
            
            # Track statistics
            recent_scores = []
            recent_max_tiles = []
            tile_achievements = {}  # Count how many times each tile is reached
            
            for episode in range(1, EPISODES + 1):
                score, steps = agent.train_episode(env)
                max_tile = int(np.max(env.game.board))
                
                # Apply learning rate decay
                if USE_LR_DECAY and episode % 10 == 0:  # Decay every 10 episodes
                    new_lr = max(MIN_LR, agent.learning_rate * LR_DECAY_RATE)
                    agent.learning_rate = new_lr
                
                recent_scores.append(score)
                recent_max_tiles.append(max_tile)
                if len(recent_scores) > 100:
                    recent_scores.pop(0)
                    recent_max_tiles.pop(0)
                
                # Track tile achievements
                tile_achievements[max_tile] = tile_achievements.get(max_tile, 0) + 1
                
                if score > best_score:
                    best_score = score
                    best_max_tile = max(best_max_tile, max_tile)
                    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                             "models", f"td_2048_best_{TUPLE_PATTERNS}.pkl")
                    agent.save_model(model_path)
                    print(f"*** NEW BEST SCORE: {best_score:,} (Episode {episode}) ***")
                
                if max_tile > best_max_tile:
                    best_max_tile = max_tile
                    print(f"*** NEW MAX TILE: {max_tile} reached! (Episode {episode}, Score: {score}) ***")
                
                if episode % 100 == 0:
                    avg_score = sum(recent_scores) / len(recent_scores)
                    avg_max_tile = sum(recent_max_tiles) / len(recent_max_tiles)
                    print(f"Episode {episode:5d} | Score: {score:6.0f} | Avg: {avg_score:6.0f} | "
                          f"Max Tile: {max_tile:5d} | Avg Tile: {avg_max_tile:5.0f} | "
                          f"Best: {best_score:6.0f} | Best Tile: {best_max_tile:5d} | "
                          f"Phase: {agent.phase} | LR: {agent.learning_rate:.4f}")
                
                # Show tile distribution every 1000 episodes
                if episode % 1000 == 0:
                    print("\nTile Achievement Distribution (last 1000 episodes):")
                    sorted_tiles = sorted(tile_achievements.keys(), reverse=True)
                    
                    # Calculate 2048 rate for decline detection
                    rate_2048 = (tile_achievements.get(2048, 0) / 1000) * 100
                    rate_4096 = (tile_achievements.get(4096, 0) / 1000) * 100
                    
                    for tile in sorted_tiles:
                        if tile >= 256:  # Only show significant tiles
                            count = tile_achievements[tile]
                            percentage = (count / 1000) * 100
                            print(f"  {tile:5d}-tile: {count:4d} times ({percentage:5.1f}%)")
                    
                    # Save periodic checkpoint with episode number
                    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                  "models", f"td_2048_ep{episode}_{TUPLE_PATTERNS}.pkl")
                    agent.save_model(checkpoint_path)
                    print(f"  [Checkpoint saved: episode {episode}]")
                    
                    # Monitor for performance decline
                    if rate_2048 > best_checkpoint_2048_rate:
                        best_checkpoint_2048_rate = rate_2048
                        checkpoint_decline_count = 0
                        print(f"  [*] New best 2048 rate: {rate_2048:.1f}%")
                    elif rate_2048 < best_checkpoint_2048_rate * 0.7:  # 30% decline
                        checkpoint_decline_count += 1
                        print(f"  [!] Performance declining ({checkpoint_decline_count}/3 warnings)")
                        if checkpoint_decline_count >= 3:
                            print(f"\n{'='*70}")
                            print(f"EARLY STOPPING: Performance declined for 3 consecutive checkpoints")
                            print(f"Best 2048 rate was: {best_checkpoint_2048_rate:.1f}%")
                            print(f"Current rate: {rate_2048:.1f}%")
                            print(f"{'='*70}\n")
                            break
                    
                    tile_achievements = {}  # Reset counter
                    print()
            
            print(f"\n{'='*70}")
            print(f"Training completed!")
            print(f"{'='*70}")
            print(f"Best Score: {best_score:,}")
            print(f"Best Max Tile: {best_max_tile}")
            print(f"Final Avg Score (last 100): {sum(recent_scores)/len(recent_scores):.0f}")
            print(f"Final Avg Max Tile (last 100): {sum(recent_max_tiles)/len(recent_max_tiles):.0f}")
            print(f"{'='*70}\n")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError during visual training: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r ../requirements.txt")