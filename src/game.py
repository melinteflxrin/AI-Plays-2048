"""
core game logic and mechanics
"""
import random
import copy


class Game2048:
    def __init__(self):
        """initialize 4x4 2048 game"""
        self.size = 4
        self.board = [[0 for _ in range(4)] for _ in range(4)]
        self.score = 0
        self.game_over = False
        
        # add two starting tiles
        self.add_random_tile()
        self.add_random_tile()
    
    def add_random_tile(self):
        """add a random tile (2 or 4) to an empty space"""
        empty_cells = []
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    empty_cells.append((i, j))
        
        if empty_cells:
            row, col = random.choice(empty_cells)
            # 90% chance for 2 and 10% chance for 4
            self.board[row][col] = 2 if random.random() < 0.9 else 4
    
    def move_left(self):
        """move all tiles left and merge them"""
        moved = False
        points = 0
        
        for i in range(4):
            # all non-zero values in this row
            row = [self.board[i][j] for j in range(4) if self.board[i][j] != 0]
            
            # merge equal values
            merged_row = []
            j = 0
            while j < len(row):
                if j < len(row) - 1 and row[j] == row[j + 1]:
                    # merge the tiles
                    merged_value = row[j] * 2
                    merged_row.append(merged_value)
                    points += merged_value
                    j += 2  # skip the next tile since we merged it
                else:
                    merged_row.append(row[j])
                    j += 1
            
            # pad with zeros to make it length 4
            merged_row += [0] * (4 - len(merged_row))
            
            # check if this row changed
            if self.board[i] != merged_row:
                moved = True
                self.board[i] = merged_row
        
        return moved, points
    
    def move_right(self):
        """move all tiles right and merge them"""
        # reverse each row, move left then reverse back
        for i in range(4):
            self.board[i] = self.board[i][::-1]
        
        moved, points = self.move_left()
        
        for i in range(4):
            self.board[i] = self.board[i][::-1]
        
        return moved, points
    
    def move_up(self):
        """move all tiles up and merge them"""
        # transpose matrix, move left transpose back
        self.transpose()
        moved, points = self.move_left()
        self.transpose()
        return moved, points
    
    def move_down(self):
        """move all tiles down and merge them"""
        # transpose matrix, move right transpose back
        self.transpose()
        moved, points = self.move_right()
        self.transpose()
        return moved, points
    
    def transpose(self):
        """transpose the board matrix"""
        self.board = [[self.board[j][i] for j in range(4)] for i in range(4)]
    
    def make_move(self, direction):
        """
        make a move in the specified direction
        """
        if self.game_over:
            return False, 0
        
        moved = False
        points = 0
        
        if direction == 'left':
            moved, points = self.move_left()
        elif direction == 'right':
            moved, points = self.move_right()
        elif direction == 'up':
            moved, points = self.move_up()
        elif direction == 'down':
            moved, points = self.move_down()
        
        if moved:
            self.score += points
            self.add_random_tile()
            
            # check if game is over
            if self.is_game_over():
                self.game_over = True
        
        return moved, points
    
    def is_game_over(self):
        """check if game is over (no more moves possible)"""
        # check for empty cells
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    return False
        
        # check for possible merges horizontally
        for i in range(4):
            for j in range(3):
                if self.board[i][j] == self.board[i][j + 1]:
                    return False
        
        # check for possible merges vertically
        for i in range(3):
            for j in range(4):
                if self.board[i][j] == self.board[i + 1][j]:
                    return False
        
        return True
    
    def reset(self):
        """reset the game"""
        self.board = [[0 for _ in range(4)] for _ in range(4)]
        self.score = 0
        self.game_over = False
        self.add_random_tile()
        self.add_random_tile()
    
    def print_board(self):
        """print the board to console (for testing)"""
        print(f"Score: {self.score}")
        print("-" * 25)
        for row in self.board:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print("    |", end="")
                else:
                    print(f"{cell:4}|", end="")
            print()
        print("-" * 25)
        if self.game_over:
            print("GAME OVER!")
        print()
