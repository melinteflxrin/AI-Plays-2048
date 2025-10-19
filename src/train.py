"""
Simplified Training Script for 2048 AI using TD Learning with N-Tuple Networks
No visualization - pure training mode for maximum speed
"""
import os
import time
import numpy as np
from collections import deque
from game_gym import Game2048Env
from td_agent import TDAgent


def train_agent(episodes=50000,
                save_frequency=100,
                tuple_patterns='8x6',
                learning_method='OTD',
                v_init=0,
                load_model=False,
                load_model_path=None,
                use_lr_decay=True,
                initial_lr=0.025,
                min_lr=0.005,
                lr_decay_rate=0.9995):
    """
    Train TD agent without visualization
    
    Args:
        episodes: Number of training episodes
        save_frequency: Save model and print stats every N episodes
        tuple_patterns: '4x6' or '8x6' tuple network configuration
        learning_method: 'OTD', 'OTC', or 'OTD+TC'
        v_init: Optimistic initialization value (0 for loaded models)
        load_model: Whether to resume from existing model
        load_model_path: Path to model file for resuming
        use_lr_decay: Enable learning rate decay
        initial_lr: Starting learning rate
        min_lr: Minimum learning rate
        lr_decay_rate: Decay rate per episode
    """
    print("\n" + "="*70)
    print("2048 AI Training - TD Learning with N-Tuple Network")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Network: {tuple_patterns}-tuple")
    print(f"  - Method: {learning_method}")
    print(f"  - Episodes: {episodes:,}")
    print(f"  - Optimistic Init: {v_init:,}")
    if load_model:
        print(f"  - Resuming from: {load_model_path}")
        print(f"  - LR Decay: {initial_lr} -> {min_lr}")
    print("="*70 + "\n")
    
    # Create environment and agent
    env = Game2048Env()
    
    # Set initial learning rate
    if use_lr_decay:
        learning_rate = initial_lr
    elif learning_method == 'OTC':
        learning_rate = 1.0
    else:
        learning_rate = 0.1
    
    agent = TDAgent(
        tuple_patterns=tuple_patterns,
        learning_method=learning_method,
        v_init=v_init,
        learning_rate=learning_rate,
        fine_tune_ratio=0.1
    )
    
    # Load existing model if specified
    if load_model and load_model_path:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), load_model_path)
        if os.path.exists(model_path):
            agent.load_model(model_path)
            print(f"[OK] Model loaded successfully from {load_model_path}\n")
        else:
            print(f"[WARNING] Model file not found at {model_path}")
            print("Starting fresh training instead.\n")
            load_model = False
    
    agent.set_total_training_episodes(episodes)
    
    # Ensure models directory exists
    os.makedirs("../models", exist_ok=True)
    
    # Initialize best scores
    # Always start from 0 so any new best score will be saved
    best_score = 0
    best_max_tile = 0
    
    # Statistics tracking
    recent_scores = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)
    tile_achievements = {}
    
    # Performance monitoring for early stopping
    best_checkpoint_2048_rate = 0
    checkpoint_decline_count = 0
    
    start_time = time.time()
    
    # Training loop
    for episode in range(1, episodes + 1):
        # Train one episode
        score, steps = agent.train_episode(env)
        max_tile = int(np.max(env.game.board))
        
        # Apply learning rate decay
        if use_lr_decay and episode % 10 == 0:
            new_lr = max(min_lr, agent.learning_rate * lr_decay_rate)
            agent.learning_rate = new_lr
        
        # Update statistics
        recent_scores.append(score)
        recent_max_tiles.append(max_tile)
        tile_achievements[max_tile] = tile_achievements.get(max_tile, 0) + 1
        
        # Save best score model
        if score > best_score:
            best_score = score
            best_max_tile = max(best_max_tile, max_tile)
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "models", f"td_2048_best_{tuple_patterns}.pkl")
            agent.save_model(model_path)
            print(f"*** NEW BEST SCORE: {best_score:,} (Episode {episode}, Max Tile: {max_tile}) ***")
        
        # Track new max tile achievements
        if max_tile > best_max_tile:
            best_max_tile = max_tile
            print(f"*** NEW MAX TILE: {max_tile} reached! (Episode {episode}, Score: {score:,}) ***")
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            avg_max_tile = sum(recent_max_tiles) / len(recent_max_tiles)
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed
            
            print(f"Ep {episode:5d} | Score: {score:6.0f} | Avg: {avg_score:6.0f} | "
                  f"Tile: {max_tile:5d} | AvgTile: {avg_max_tile:5.0f} | "
                  f"Best: {best_score:6.0f}/{best_max_tile:5d} | "
                  f"LR: {agent.learning_rate:.4f} | {eps_per_sec:.1f} ep/s")
        
        # Detailed statistics every 1000 episodes
        if episode % 1000 == 0:
            print("\nTile Achievement Distribution (last 1000 episodes):")
            sorted_tiles = sorted(tile_achievements.keys(), reverse=True)
            
            # Calculate achievement rates
            rate_2048 = (tile_achievements.get(2048, 0) / 1000) * 100
            rate_4096 = (tile_achievements.get(4096, 0) / 1000) * 100
            rate_8192 = (tile_achievements.get(8192, 0) / 1000) * 100
            
            for tile in sorted_tiles:
                if tile >= 256:
                    count = tile_achievements[tile]
                    percentage = (count / 1000) * 100
                    print(f"  {tile:5d}-tile: {count:4d} times ({percentage:5.1f}%)")
            
            # Monitor for performance decline (early stopping)
            if rate_2048 > best_checkpoint_2048_rate:
                best_checkpoint_2048_rate = rate_2048
                checkpoint_decline_count = 0
                print(f"  [*] New best 2048 rate: {rate_2048:.1f}%")
            elif rate_2048 < best_checkpoint_2048_rate * 0.7:
                checkpoint_decline_count += 1
                print(f"  [!] Performance declining ({checkpoint_decline_count}/3 warnings)")
                if checkpoint_decline_count >= 3:
                    print(f"\n{'='*70}")
                    print(f"EARLY STOPPING: Performance declined for 3 consecutive checkpoints")
                    print(f"Best 2048 rate was: {best_checkpoint_2048_rate:.1f}%")
                    print(f"Current rate: {rate_2048:.1f}%")
                    print(f"{'='*70}\n")
                    break
            
            # Reset tile counter for next 1000 episodes
            tile_achievements = {}
            print()
    
    # Training complete - print final statistics
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n{'='*70}")
    print(f"Training Completed!")
    print(f"{'='*70}")
    print(f"Best Score: {best_score:,}")
    print(f"Best Max Tile: {best_max_tile}")
    print(f"Final Avg Score (last 100): {sum(recent_scores)/len(recent_scores):.0f}")
    print(f"Final Avg Max Tile (last 100): {sum(recent_max_tiles)/len(recent_max_tiles):.0f}")
    print(f"Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Total Episodes: {episode}")
    print(f"{'='*70}\n")
    
    return agent


if __name__ == "__main__":
    # ===================================================================
    # TRAINING CONFIGURATION
    # ===================================================================
    
    # Number of training episodes
    EPISODES = 50000
    
    # Network configuration
    TUPLE_PATTERNS = '8x6'
    
    # Learning method
    LEARNING_METHOD = 'OTD'
    
    # Optimistic initialization (only for fresh training)
    V_INIT = 0
    
    # Resume from existing model
    LOAD_MODEL = True
    LOAD_MODEL_PATH = "models/td_2048_best_8x6.pkl"
    
    # Learning rate settings
    USE_LR_DECAY = True
    INITIAL_LR = 0.025
    MIN_LR = 0.005
    LR_DECAY_RATE = 0.9995
    
    # Save frequency
    SAVE_FREQUENCY = 100
    
    # ===================================================================
    
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
    print("="*70 + "\n")
    
    try:
        agent = train_agent(
            episodes=EPISODES,
            save_frequency=SAVE_FREQUENCY,
            tuple_patterns=TUPLE_PATTERNS,
            learning_method=LEARNING_METHOD,
            v_init=V_INIT,
            load_model=LOAD_MODEL,
            load_model_path=LOAD_MODEL_PATH,
            use_lr_decay=USE_LR_DECAY,
            initial_lr=INITIAL_LR,
            min_lr=MIN_LR,
            lr_decay_rate=LR_DECAY_RATE
        )
        
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Progress has been saved in checkpoint files.")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
