"""
Test script for the 2048 RL environment
"""
import sys
import os

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from game_env import Game2048Env

def test_environment():
    """Test the 2048 environment with random actions"""
    print("Testing 2048 RL Environment...")
    print("=" * 40)
    
    # Create environment
    env = Game2048Env()
    
    # Reset environment
    observation, info = env.reset()
    print(f"Initial state:")
    print(f"Score: {info['score']}")
    env.render()
    
    # Play a few random moves
    for step in range(10):
        # Choose random action
        action = env.action_space.sample()
        action_name = env.action_to_direction[action]
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"Action: {action_name} ({action})")
        print(f"Reward: {reward:.2f}")
        print(f"Score: {info['score']}")
        print(f"Moved: {info['moved']}")
        
        env.render()
        
        if terminated:
            print("\nGame Over!")
            break
        
        if truncated:
            print("\nEpisode truncated!")
            break
    
    print(f"\nFinal Score: {info['score']}")
    print("Test completed!")

if __name__ == "__main__":
    test_environment()