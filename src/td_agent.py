"""
TD Agent with Optimistic Temporal Difference Learning for 2048
based on research (Hung Guei et al., 2023)

replaces the DQN agent with a more effective approach:
- N-tuple networks instead of deep neural networks
- TD learning instead of Q-learning (better for 2048)
"""
import numpy as np
import copy
from collections import defaultdict
from ntuple_network import NTupleNetwork


class TDAgent:
    """
    temporal difference learning agent for 2048
    
    three learning methods:
    1. OTD (Optimistic TD): fixed learning rate with optimistic initialization
    2. OTC (Optimistic TC): temporal coherence with adaptive learning rates 
        - TC -> slower learning for oscillating patterns
    3. OTD+TC: hybrid approach (OTD first, then TC fine-tuning)
    
    performance (from research paper):
    - OTC + 4x6-tuple: ~280k avg score, 5.3% reach 32768 tile
    - OTD+TC + 8x6-tuple: ~371k avg score, 22% reach 32768 tile
    """
    
    def __init__(self, 
                 tuple_patterns='4x6',
                 learning_method='OTD+TC',
                 v_init=320000,
                 learning_rate=0.1,
                 fine_tune_ratio=0.1,
                 trace_lambda=0.0):
        """
        args:
            tuple_patterns: '4x6' (faster) or '8x6' (better performance)
            learning_method: 'OTD', 'OTC', or 'OTD+TC'
            v_init: optimistic initial value
            learning_rate: how much to adjust values when discovering an error (0.1 for OTD, 1.0 for OTC)
            fine_tune_ratio: proportion of training for TC fine-tuning in OTD+TC
            trace_lambda: how much to look behind when learning
        """
        self.tuple_patterns = tuple_patterns
        self.learning_method = learning_method
        self.v_init = v_init
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.fine_tune_ratio = fine_tune_ratio
        self.trace_lambda = trace_lambda
        
        # initialize N-tuple network
        self.network = NTupleNetwork(
            tuple_patterns=tuple_patterns,
            n_values=16,  # up to 32768 tile
            v_init=v_init
        )
        
        # temporal coherence parameters (for OTC and OTD+TC fine-tuning)
        self.coherence_params = [defaultdict(dict) for _ in range(self.network.n_tuples)]
        
        # training state
        self.episodes_trained = 0
        self.phase = 'exploration'  # 'exploration' or 'fine-tuning'
        self.total_training_episodes = None
        
        # learning rate schedule for OTD
        self.lr_schedule = {
            0.50: 0.01,   # at 50% training, reduce to 0.01
            0.75: 0.001,  # at 75% training, reduce to 0.001
        }
        
        # statistics
        self.stats = {
            'positive_td_errors': 0,
            'negative_td_errors': 0,
            'total_updates': 0
        }
        
        print(f"\n{'='*60}")
        print(f"TD Agent Initialized")
        print(f"{'='*60}")
        print(f"Network: {tuple_patterns}-tuple")
        print(f"Learning Method: {learning_method}")
        print(f"Optimistic Init: {v_init:,}")
        print(f"Learning Rate: {learning_rate}")
        if learning_method == 'OTD+TC':
            print(f"Fine-tune Ratio: {fine_tune_ratio*100}%")
        print(f"{'='*60}\n")
    
    def set_total_training_episodes(self, total):
        """set total training episodes"""
        self.total_training_episodes = total
    
    def _update_learning_rate(self):
        """update learning rate based on training progress (for OTD method)"""
        if self.learning_method == 'OTD' and self.total_training_episodes:
            progress = self.episodes_trained / self.total_training_episodes
            
            for threshold, new_lr in sorted(self.lr_schedule.items()):
                if progress >= threshold and self.learning_rate > new_lr:
                    self.learning_rate = new_lr
                    print(f"\n[Episode {self.episodes_trained}] Learning rate reduced to {new_lr}")
                    break
    
    def _update_phase(self):
        """switch from exploration to fine-tuning phase (for OTD+TC method)"""
        if self.learning_method == 'OTD+TC' and self.total_training_episodes:
            progress = self.episodes_trained / self.total_training_episodes
            fine_tune_start = 1.0 - self.fine_tune_ratio
            
            if progress >= fine_tune_start and self.phase == 'exploration':
                self.phase = 'fine-tuning'
                self.learning_rate = 1.0  # TC learning uses α=1.0
                print(f"\n{'='*60}")
                print(f"[Episode {self.episodes_trained}] Switching to TC Fine-tuning Phase")
                print(f"{'='*60}\n")
    
    def get_afterstate(self, board, action):
        """
        we should get the afterstate: 
        - board after move but before random tile
        (returns the deterministic result of the player's action)

        for simplicity, we will work with the state after random tile
        
        args:
            action: 0=up, 1=down, 2=left, 3=right
            
        returns:
            afterstate_board: board after move
            reward: points earned from merging
            valid: if the move was valid
        """
        # create a copy to simulate the move
        from game import Game2048
        temp_game = Game2048()
        temp_game.board = copy.deepcopy(board)
        temp_game.score = 0
        
        action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        moved, points = temp_game.make_move(action_map[action])
        
        if moved:
            # remove the random tile that was added
            # we want the afterstate (before random tile)
            # find and remove the last added tile
            pass  # for simplicity -> work with the state after random tile
        
        return temp_game.board, points, moved
    
    def choose_action(self, board, valid_actions=None, greedy=True):
        """
        choose action using greedy policy based on afterstate values
        
        implements the policy from equation (10) in the paper:
        
        args:
            board: current board state
            valid_actions: list of valid actions (None = try all)
            greedy: always choose best action (True for training)
            
        returns:
            action: best action (0=up, 1=down, 2=left, 3=right)
        """
        if valid_actions is None:
            valid_actions = [0, 1, 2, 3]
        
        best_action = None
        best_value = -float('inf')
        
        # evaluate each action
        for action in valid_actions:
            afterstate_board, reward, valid = self.get_afterstate(board, action)
            
            if not valid:
                continue
            
            # calculate action value: immediate reward + afterstate value
            afterstate_value = self.network.evaluate(afterstate_board)
            action_value = reward + afterstate_value
            
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
        # if no valid action found, return random
        if best_action is None:
            best_action = np.random.choice(valid_actions)
        
        return best_action
    
    def train_episode(self, env):
        """
        train for one episode using TD learning with afterstate framework
        
        implements the core training loop from the paper.
        each episode:
        1. agent takes action -> gets afterstate
        2. environment adds random tile -> gets next state
        3. calculate TD error and update network
        4. repeat until game over
        
        args:
            env: Game2048Env instance
            
        returns:
            episode_score: final score of the episode
            episode_steps: number of moves made
        """
        observation, info = env.reset()
        board = env.game.board
        
        episode_history = []  # store (afterstate, reward) for backward update
        episode_score = 0
        episode_steps = 0
        
        while True:
            # 1. choose action and get afterstate
            action = self.choose_action(board)
            
            # store current board for update
            current_board = copy.deepcopy(board)
            
            # 2. take action in environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_board = env.game.board
            
            # get afterstate (board after our move, before random tile)
            afterstate_board, move_reward, valid = self.get_afterstate(current_board, action)
            
            if not valid:
                # invalid move, skip
                continue
            
            # store for backward update
            episode_history.append((afterstate_board, move_reward, next_board, terminated))
            
            episode_score = info['score']
            episode_steps += 1
            
            # update board for next iteration
            board = next_board
            
            # check if game ended
            if terminated or truncated:
                break
        
        # backward update: update all afterstates in the episode
        self._backward_update(episode_history)
        
        # update training state
        self.episodes_trained += 1
        self._update_learning_rate()
        self._update_phase()
        
        return episode_score, episode_steps
    
    def _backward_update(self, episode_history):
        """
        update network weights using backward update strategy
        
        processes the episode from the end to the beginning

        args:
            episode_history: list of (afterstate, reward, next_state, done) tuples
        """
        # process from end to beginning
        for i in range(len(episode_history) - 1, -1, -1):
            afterstate, reward, next_state, done = episode_history[i]
            
            # current afterstate value
            current_value = self.network.evaluate(afterstate)
            
            # calculate target value
            if i == len(episode_history) - 1 or done:
                # terminal afterstate
                target_value = 0.0
            else:
                # get next afterstate value
                next_afterstate = episode_history[i + 1][0]
                next_reward = episode_history[i + 1][1]
                next_value = self.network.evaluate(next_afterstate)
                target_value = next_reward + next_value
            
            # calculate TD error
            td_error = target_value - current_value
            
            # update statistics
            if td_error > 0:
                self.stats['positive_td_errors'] += 1
            else:
                self.stats['negative_td_errors'] += 1
            self.stats['total_updates'] += 1
            
            # update network based on learning method
            if self.phase == 'fine-tuning' or self.learning_method == 'OTC':
                # use Temporal Coherence learning
                self.network.update_with_coherence(
                    afterstate, 
                    td_error, 
                    self.coherence_params
                )
            else:
                # use standard TD learning
                self.network.update(
                    afterstate,
                    td_error,
                    self.learning_rate
                )
    
    def get_exploration_stats(self):
        """get statistics about exploration (for monitoring)"""
        total = self.stats['total_updates']
        if total == 0:
            return {'positive_ratio': 0.0, 'negative_ratio': 0.0}
        
        return {
            'positive_ratio': self.stats['positive_td_errors'] / total,
            'negative_ratio': self.stats['negative_td_errors'] / total,
            'total_updates': total
        }
    
    def save_model(self, filepath):
        """save the trained model"""
        self.network.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """load a trained model"""
        self.network.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_network_statistics(self):
        """get network statistics for monitoring"""
        return self.network.get_statistics()
