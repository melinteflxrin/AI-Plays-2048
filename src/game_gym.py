import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import Game2048


class Game2048Env(gym.Env):
    """
    Gymnasium environment for 2048 game
    """
    
    def __init__(self):
        super().__init__()
        
        self.game = Game2048()
        
        # actions -> 4 possible moves
        # 0 = up, 1 = down, 2 = left, 3 = right
        self.action_space = spaces.Discrete(4)
        
        # observation space -> 4x4 grid
        # normalize tile values using log2 (so 2->1, 4->2, 8->3, etc.)
        self.observation_space = spaces.Box(
            low=0,
            high=15,
            shape=(4, 4),
            dtype=np.float32
        )
        
        # map actions to game directions
        self.action_to_direction = {
            0: 'up',
            1: 'down', 
            2: 'left',
            3: 'right'
        }
    
    def _get_observation(self):
        """
        convert the 4x4 game board to an observation
        log2 to normalize tile values
        """
        obs = np.zeros((4, 4), dtype=np.float32)
        
        for i in range(4):
            for j in range(4):
                if self.game.board[i][j] > 0:
                    # normalize using log2
                    obs[i][j] = np.log2(self.game.board[i][j])
                else:
                    obs[i][j] = 0
        
        return obs
    
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
        one step in the environment
        """
        # convert action number to direction
        direction = self.action_to_direction[action]
        
        # store old score for reward calculation
        old_score = self.game.score
        
        # make the move
        moved, points = self.game.make_move(direction)
        
        # calculate reward
        reward = 0.0
        if moved:
            reward = points * 0.01 + 0.1  # points reward + small move bonus
        else:
            reward = -0.1  # small penalty for invalid moves

        # get new observation
        observation = self._get_observation()

        # check if game is over
        terminated = self.game.game_over
        truncated = False
        
        # info dictionary
        info = {
            "score": self.game.score,
            "moved": moved,
            "points_gained": points
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        """display the game state"""
        if mode == "human":
            self.game.print_board()
    
    def close(self):
        """clean up resources"""
        pass