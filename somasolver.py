from locale import normalize
import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Rotation matrices
Rx = np.array([[ 1, 0, 0],
               [ 0, 0,-1],
               [ 0, 1, 0]])

Ry = np.array([[ 0, 0, 1],
               [ 0, 1, 0],
               [-1, 0, 0]])

Rz = np.array([[ 0,-1, 0],
               [ 1, 0, 0],
               [ 0, 0, 1]])

class Block:
    def __init__(self, coords, name):
        self.coord_len = len(coords)
        self.coords = np.array(coords).T
        self.origin = np.array([0,0,0])
        self.unique_rotations = self.generate_unique_rotations()
        # self.direction = np.array([[1],[0],[0]])
        self.bounding_box = np.max(self.coords, axis=1)
        self.name = name

    def __len__(self):
        return self.coord_len

    def __getitem__(self, i):
        return self.coords[:,i]
    
    def set_coords(self, new_coords):
        self.coords = new_coords
        self.bounding_box = np.max(self.coords, axis=1)

    def set_origin(self, origin_coords):
        self.origin = np.array(origin_coords)

    def roll(self):
        self.coords = Rx @ self.coords
        # self.direction = Rx @ self.direction
    
    def turn(self):
        self.coords = Rz @ self.coords
        # self.direction = Rz @ self.direction
    
    @staticmethod
    def equivalent_coords(coord1, coord2):
        if coord1.shape != coord2.shape:
            return False
        n = coord1.shape[1]
        for i in range(n):
            c1 = coord1[:,i]
            c1_in_coord2 = False
            for j in range(n):
                c2 = coord2[:,j]
                if (c1 == c2).all():
                    c1_in_coord2 = True
            if not c1_in_coord2:
                return False
        return True
                

    def _get_all_rotations(self):
        """Return a list of the 24 3d rotations"""
        rotations = []
        for cycle in range(2):
            for step in range(3): 
                self.roll()
                rotations.append(copy(self.coords))
                for i in range(3):
                    self.turn()
                    rotations.append(copy(self.coords))
            self.roll()
            self.turn()
            self.roll()
        return rotations
    
    def normalize_rotations(self, rotations):
        normalized_rotations = []
        for coords in rotations:
            normalized_rotations.append(coords - np.min(coords, axis=1)[:,None])
        return normalized_rotations

    def generate_unique_rotations(self):
        """Return a list of unique 3d rotations with duplicates due to symmetry removed"""
        all_rotations = self._get_all_rotations()
        normalized_rotations = self.normalize_rotations(all_rotations)
        unique_rotations = []
        for new_rot in normalized_rotations:
            is_duplicate = False
            for unique_rot in unique_rotations:
                # if (new_rot == unique_rot).all():
                if self.equivalent_coords(new_rot, unique_rot):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_rotations.append(new_rot)
        return unique_rotations


    def rotations(self):
        for rot in self.unique_rotations:
            self_copy = deepcopy(self)
            self_copy.set_coords(rot)
            yield(self_copy)

    # def rotations(self):
    #     """Generator for the 24 unique 3d rotations"""
    #     for cycle in range(2):
    #         for step in range(3): 
    #             self.roll()
    #             yield(self.coords)
    #             for i in range(3):
    #                 self.turn()
    #                 yield(self.coords)
    #         self.roll()
    #         self.turn()
    #         self.roll()

class Box:
    def __init__(self):
        self.filled = np.zeros((3, 3, 3), dtype=int)
    
    @property
    def is_filled(self):
        return (self.filled == 1).all()
    
    @property
    def num_filled(self):
        return np.sum(self.filled)
    
    def coord_is_filled(self, i, j, k):
        return self.filled[i, j, k]

    def validate_placement(self, block, origin):
        for offset in block:
            cube_coord = origin + offset
            if (cube_coord < 0).any() or (cube_coord > 2).any():
                raise IndexError('Block placement is out of bounds.')
            if self.filled[tuple(cube_coord)]:
                raise IndexError('Space is already filled.')

    def place(self, block, origin):
        self.validate_placement(block, origin)
        for offset in block:
            cube_coord = origin + offset
            self.filled[tuple(cube_coord)] = 1

class GameState:
    def __init__(self, block_list):
        self.block_candidates = [Block(b, f'B{i}') for i, b in enumerate(block_list)]
        self.placed_blocks = []
        self.box = Box()
    
    @property
    def is_solved(self):
        return self.box.is_filled
    
    def print(self):
        print(f'Places filled: {self.box.num_filled}')
        print('Candidate blocks: ', end='')
        for bc in self.block_candidates:
            print(bc.name, end=' ')
        print('\nPlaced blocks: ', end='')
        for pb in self.placed_blocks:
            print(pb.name, end=' ')
        print()

def solve(game):
    game.print()
    if not game.is_solved and len(game.block_candidates) > 0:
        placement_block = game.block_candidates.pop()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if game.box.coord_is_filled(i, j, k):
                        continue

                    placement_block.set_origin([i, j, k])
                    for block_rotation in placement_block.rotations():
                        game_copy = deepcopy(game)
                        try:
                            game_copy.box.place(block_rotation, [i, j, k])
                            game_copy.placed_blocks.append(block_rotation)
                            # placement_block_copy = deepcopy(placement_block)
                            # game_copy.box.place(placement_block_copy, [i, j, k])
                            # game_copy.placed_blocks.append(placement_block_copy)
                            game_copy = solve(game_copy)
                            if game_copy.is_solved:
                                return game_copy
                        except IndexError:
                            pass
    return game

def plot_solution(game):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title('Cube solution')
    for c, b in enumerate(game.placed_blocks):
        coords = b.coords + b.origin[:,None]
        ax.scatter(coords[0], coords[1], coords[2], color=f'C{c}', s=200, alpha=1)
        for cube1 in b:
            for cube2 in b:
                c1 = cube1 + b.origin
                c2 = cube2 + b.origin
                if np.sum((c1 - c2)**2) == 1:
                    ax.plot3D((c1[0], c2[0]), (c1[1], c2[1]), (c1[2], c2[2]), color=f'C{c}', lw=10)
    plt.show()


B1 = [[0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [2, 1, 0]]

B2 = [[0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [1, 1, 0]]

B3 = [[0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 1, 1]]

B4 = [[0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [1, 1, 0],
      [1, 1, 1]]

B5 = [[0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [1, 0, 1],
      [2, 0, 1]]

B6 = [[0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [1, 1, 0],
      [2, 0, 1]]

block_list = [B1, B2, B3, B4, B5, B6]

def main():
    game = GameState(block_list)
    game_solved = solve(game)
    if game_solved.is_solved:
        print('yay!')
        print(len(game_solved.placed_blocks))
        for b in game_solved.placed_blocks:
            print(b.name)
            print(b.coords + b.origin[:,None])
        plot_solution(game_solved)
    else:
        print('Found no solution :\'(')

if __name__ == "__main__":
    main()
