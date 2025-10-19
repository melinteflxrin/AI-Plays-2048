import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import Game2048
import copy


class Game2048Env(gym.Env):
    """
    gymnasium environment for 2048 game
    
    afterstate learning framework:
    - state and afterstate representations
    - tracks board before and after random tile placement
    - for TD learning
    """
    
    def __init__(self):
        super().__init__()
        
        self.game = Game2048()
        
        # actions -> 4 possible moves
        # 0 = up, 1 = down, 2 = left, 3 = right
        self.action_space = spaces.Discrete(4)
        
        # observation space -> 4x4 grid
        # for TD learning, work directly with board values (not log2)
        # allows N-tuple network to extract features properly
        self.observation_space = spaces.Box(
            low=0,
            high=131072,  # up to 131072 tile (not reaching here anyways)
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
        
        # track afterstate (board after move, before random tile)
        self.last_afterstate = None
    
    def _get_observation(self):
        """
        convert 4x4 game board to an observation
        
        for TD learning with N-tuple networks, return the raw board
        (not log2 normalized) so the network can properly extract features
        """
        obs = np.array(self.game.board, dtype=np.int32)
        return obs
    
    def get_afterstate(self, action):
        """
        get the afterstate: board after move but before random tile
        
        necessary for afterstate learning framework:
        returns the deterministic result of the player's action.
        
        args:
            action: 0=up, 1=down, 2=left, 3=right
            
        returns:
            afterstate_board: Board after move (before random tile)
            reward: points earned from merging
            valid: if the move was valid
        """
        # create a temporary game to simulate the move
        temp_game = Game2048()
        temp_game.board = copy.deepcopy(self.game.board)
        temp_game.score = self.game.score
        temp_game.game_over = self.game.game_over
        
        direction = self.action_to_direction[action]
        moved, points = temp_game.make_move(direction)
        
        if not moved:
            return None, 0, False
        
        # the afterstate is the board before add_random_tile was called
        # need to simulate without the random tile
        temp_game2 = Game2048()
        temp_game2.board = copy.deepcopy(self.game.board)
        temp_game2.score = self.game.score
        temp_game2.game_over = self.game.game_over
        
        # manually perform the move without adding random tile
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
        take one step in the environment
        
        for TD learning:
        - return actual board state (not log2)
        - track afterstate for learning
        - provide detailed info for TD updates
        """
        # store afterstate before random tile
        afterstate_board, move_reward, valid = self.get_afterstate(action)
        
        # convert action number to direction
        direction = self.action_to_direction[action]
        
        # make the move (this adds random tile)
        moved, points = self.game.make_move(direction)
        
        # calculate REWARD
        # for TD learning, the reward is just the points earned
        # (no artificial bonuses/penalties)
        reward = float(points) if moved else 0.0
        
        # get new observation (board with random tile)
        observation = self._get_observation()
        
        # check if game is over
        terminated = self.game.game_over
        truncated = False
        
        # store afterstate for potential use
        if valid:
            self.last_afterstate = afterstate_board
        
        # info dictionary for TD learning
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