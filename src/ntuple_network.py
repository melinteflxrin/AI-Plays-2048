"""
N-Tuple Network for 2048

implements a lookup table-based feature extraction network
that significantly outperforms deep neural networks for 2048
"""
import numpy as np
from collections import defaultdict
import pickle


class NTupleNetwork:
    """
    N-Tuple Network with symmetric sampling (8 isomorphic transformations)
    
    advantages over DQN:
    - faster training (simple lookup vs neural network)
    - better generalization -> symmetric sampling
    """
    
    def __init__(self, tuple_patterns='4x6', n_values=16, v_init=0.0):
        """
        args:
            tuple_patterns: '4x6' or '8x6' tuple configuration
            n_values: number of distinct tile values (16 = up to 32768 tile)
            v_init: initial value for optimistic initialization
        """
        self.n_values = n_values
        self.v_init = v_init
        
        # define tuple patterns based on research
        if tuple_patterns == '4x6':
            self.tuples = self._get_4x6_tuples()
        elif tuple_patterns == '8x6':
            self.tuples = self._get_8x6_tuples()
        else:
            raise ValueError(f"Unknown tuple pattern: {tuple_patterns}")
        
        self.n_tuples = len(self.tuples)
        
        # initialize lookup tables for each tuple
        # using defaultdict for automatic initialization 
        # -> add new entries of patterns into the dict
        self.weights = [defaultdict(lambda: v_init / self.n_tuples) 
                       for _ in range(self.n_tuples)]
        
        print(f"Initialized {tuple_patterns}-tuple network with {self.n_tuples} tuples")
        print(f"Optimistic initialization: {v_init}")
    
    def _get_4x6_tuples(self):
        """
        Yeh's 4x6-tuple network configuration (Figure 4 from paper)
        Used in the research, achieves ~280k average score with OTC
        """
        return [
            # row patterns
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],  # top-left corner
            [(0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3)],  # top-right corner
            # column patterns
            [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1)],  # left column
            [(2, 2), (3, 2), (2, 3), (3, 3), (1, 2), (1, 3)],  # bottom-right corner
        ]
    
    def _get_8x6_tuples(self):
        """
        Matsuzaki's 8x6-tuple network configuration (Figure 5 from paper)
        achieves ~371k average score with OTD+TC
        """
        return [
            # row patterns (horizontal)
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
            [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],
            [(0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2)],
            [(0, 3), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            # additional patterns for better coverage
            [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
            [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)],
            [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)],
            [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)],
        ]
    
    def _board_to_tuple(self, board, positions):
        """
        extract tuple feature from board at given positions
        
        returns:
            tuple: feature values at specified positions
        """
        return tuple(int(board[i][j]) for i, j in positions)
    
    def _get_isomorphic_boards(self, board):
        """
        make all 8 isomorphic transformations (rotations + mirrors)
        
        the key to symmetric sampling:
        - 4 rotations (0°, 90°, 180°, 270°)
        - each rotation can be mirrored
        
        returns:
            list: 8 transformed boards
        """
        boards = []
        current = np.array(board)
        
        # 4 rotations
        for _ in range(4):
            boards.append(current.tolist())
            boards.append(np.fliplr(current).tolist())  # mirror horizontally
            current = np.rot90(current)
        
        return boards
    
    def evaluate(self, board):
        """
        evaluate board state using all tuples and symmetric sampling
        -> how good are the patterns of the board?
        core value function V(s) that sums all feature weights
        
        args:
            board: 4x4 board state (list of lists or numpy array)
            
        returns:
            float: estimated value of the board state
        """
        total_value = 0.0
        
        # get all 8 isomorphic boards
        iso_boards = self._get_isomorphic_boards(board)

        # for each tuple, sum values across all isomorphic transformations
        for tuple_idx, positions in enumerate(self.tuples):
            for iso_board in iso_boards:
                feature = self._board_to_tuple(iso_board, positions)
                total_value += self.weights[tuple_idx][feature]
        
        return total_value
    
    def update(self, board, delta, learning_rate=0.1):
        """
        update lookup table using TD error
        
        args:
            board: 4x4 board state
            delta: TD error (target - current_value)
            learning_rate
        """
        # distribute update equally across all tuples
        update_amount = (learning_rate / self.n_tuples) * delta

        # get all 8 isomorphic boards
        iso_boards = self._get_isomorphic_boards(board)

        # update weights for each tuple and each board orientation
        for tuple_idx, positions in enumerate(self.tuples):
            for iso_board in iso_boards:
                feature = self._board_to_tuple(iso_board, positions)
                self.weights[tuple_idx][feature] += update_amount
    
    def update_with_coherence(self, board, delta, coherence_params):
        """
        update using Temporal Coherence (TC) learning with adaptive learning rate
        
        args:
            board: 4x4 board state
            delta: TD error
            coherence_params: dict with 'psi' and 'phi' for each feature

        psi - sum of errors (the net direction of this pattern)
        phi - sum of absolute errors (how much this pattern has been updated)
        """
        iso_boards = self._get_isomorphic_boards(board)
        
        for tuple_idx, positions in enumerate(self.tuples):
            for iso_board in iso_boards:
                feature = self._board_to_tuple(iso_board, positions)
                
                # get coherence parameters for this feature
                if feature not in coherence_params[tuple_idx]:
                    coherence_params[tuple_idx][feature] = {'psi': 0.0, 'phi': 0.0}
                
                params = coherence_params[tuple_idx][feature]
                
                # update psi and phi (equation 16 from paper)
                params['psi'] += delta
                params['phi'] += abs(delta)

                # calculate coherence (equation 15 from paper)
                if params['phi'] > 0:
                    coherence = abs(params['psi']) / params['phi']
                else:
                    coherence = 1.0

                # update weight with coherence modulation
                update_amount = (coherence / self.n_tuples) * delta
                self.weights[tuple_idx][feature] += update_amount
    
    def save(self, filepath):
        """save the network weights to file"""
        save_data = {
            'weights': [{k: v for k, v in w.items()} for w in self.weights],
            'tuples': self.tuples,
            'n_tuples': self.n_tuples,
            'n_values': self.n_values,
            'v_init': self.v_init
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Network saved to {filepath}")
    
    def load(self, filepath):
        """load network weights from file"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.tuples = save_data['tuples']
        self.n_tuples = save_data['n_tuples']
        self.n_values = save_data['n_values']
        self.v_init = save_data['v_init']
        
        # reconstruct defaultdicts
        self.weights = [defaultdict(lambda: 0.0) for _ in range(self.n_tuples)]
        for i, w in enumerate(save_data['weights']):
            self.weights[i].update(w)
        
        print(f"Network loaded from {filepath}")
    
    def get_statistics(self):
        """get statistics about the network"""
        total_features = sum(len(w) for w in self.weights)
        avg_value = np.mean([np.mean(list(w.values())) for w in self.weights if len(w) > 0])
        
        return {
            'total_features': total_features,
            'avg_feature_value': avg_value,
            'n_tuples': self.n_tuples
        }
