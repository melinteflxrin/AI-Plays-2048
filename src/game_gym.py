import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import Game2048
import copy


class Game2048Env(gym.Env):
    """
    Gymnasium environment for 2048 game
    
    Enhanced to support afterstate learning framework:
    - Provides both state and afterstate representations
    - Tracks board before and after random tile placement
    - Optimized for TD learning (not Q-learning)
    """
    
    def __init__(self):
        super().__init__()
        
        self.game = Game2048()
        
        # actions -> 4 possible moves
        # 0 = up, 1 = down, 2 = left, 3 = right
        self.action_space = spaces.Discrete(4)
        
        # observation space -> 4x4 grid
        # For TD learning, we work directly with board values (not log2)
        # This allows N-tuple network to extract features properly
        self.observation_space = spaces.Box(
            low=0,
            high=131072,  # Up to 131072-tile (though rare)
            shape=(4, 4),
            dtype=np.int32
        )
        
        # map actions to game directions
        self.action_to_direction = {
            0: 'up',
            1: 'down', 
            2: 'left',
            3: 'right'
        }
        
        # Track afterstate (board after move, before random tile)
        self.last_afterstate = None
    
    def _get_observation(self):
        """
        Convert the 4x4 game board to an observation
        
        For TD learning with N-tuple networks, we return the raw board
        (not log2 normalized) so the network can properly extract features
        """
        obs = np.array(self.game.board, dtype=np.int32)
        return obs
    
    def get_afterstate(self, action):
        """
        Get the afterstate: board after move but before random tile
        
        This is critical for afterstate learning framework.
        Returns the deterministic result of the player's action.
        
        Args:
            action: 0=up, 1=down, 2=left, 3=right
            
        Returns:
            afterstate_board: Board after move (before random tile)
            reward: Points earned from merging
            valid: Whether the move was valid
        """
        # Create a temporary game to simulate the move
        temp_game = Game2048()
        temp_game.board = copy.deepcopy(self.game.board)
        temp_game.score = self.game.score
        temp_game.game_over = self.game.game_over
        
        direction = self.action_to_direction[action]
        moved, points = temp_game.make_move(direction)
        
        if not moved:
            return None, 0, False
        
        # The afterstate is the board before add_random_tile was called
        # We need to simulate without the random tile
        temp_game2 = Game2048()
        temp_game2.board = copy.deepcopy(self.game.board)
        temp_game2.score = self.game.score
        temp_game2.game_over = self.game.game_over
        
        # Manually perform the move without adding random tile
        if direction == 'left':
            _, pts = temp_game2.move_left()
        elif direction == 'right':
            _, pts = temp_game2.move_right()
        elif direction == 'up':
            _, pts = temp_game2.move_up()
        elif direction == 'down':
            _, pts = temp_game2.move_down()
        
        afterstate_board = np.array(temp_game2.board, dtype=np.int32)
        return afterstate_board, pts, True
    
    def reset(self, seed=None, options=None):
        """reset the game to start a new episode"""
        super().reset(seed=seed)
        
        self.game.reset()
        
        # get initial observation
        observation = self._get_observation()
        
        # return observation and info (required by Gymnasium)
        info = {"score": self.game.score}
        
        return observation, info
    
    def step(self, action):
        """
        Take one step in the environment
        
        Enhanced for TD learning:
        - Returns actual board state (not log2)
        - Tracks afterstate for learning
        - Provides detailed info for TD updates
        """
        # Store afterstate before random tile
        afterstate_board, move_reward, valid = self.get_afterstate(action)
        
        # Convert action number to direction
        direction = self.action_to_direction[action]
        
        # Make the move (this adds random tile)
        moved, points = self.game.make_move(direction)
        
        # Calculate reward
        # For TD learning, the reward is simply the points earned
        # (no artificial bonuses/penalties)
        reward = float(points) if moved else 0.0
        
        # Get new observation (board with random tile)
        observation = self._get_observation()

        # Check if game is over
        terminated = self.game.game_over
        truncated = False
        
        # Store afterstate for potential use
        if valid:
            self.last_afterstate = afterstate_board
        
        # Enhanced info dictionary for TD learning
        info = {
            "score": self.game.score,
            "moved": moved,
            "points_gained": points,
            "afterstate": afterstate_board if valid else None,
            "max_tile": int(np.max(self.game.board))
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        """display the game state"""
        if mode == "human":
            self.game.print_board()
    
    def close(self):
        """clean up resources"""
        pass