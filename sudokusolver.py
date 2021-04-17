import numpy as np
import copy


class Points:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.ij = i, j

    def iterate(self):
        self.j += 1
        if self.j == 9:
            self.j = 0
            self.i += 1
            if self.i == 9:
                self.i = 0
        self.ij = self.i, self.j

    def box_iter(self):
        self.j += 1
        if self.j % 3 == 0:
            self.j -= 3
            self.i += 1
            if self.i % 3 == 0:
                self.i -= 3
        self.ij = self.i, self.j

    def line_iter(self):
        self.j += 1
        if self.j == 9:
            self.j = 0
        self.ij = self.i, self.j

    def col_iter(self):
        self.i += 1
        if self.i == 9:
            self.i = 0
        self.ij = self.i, self.j


class SudokuBoard:
    def __init__(self):
        self.board = np.full((9, 9), {1, 2, 3, 4, 5, 6, 7, 8, 9}, dtype=set)

        self.solved = False
        self.solvable = True

    def import_board(self, file_name):
        # File should contain a single sudoku board with 0 indicating an empty box
        f = open(file_name, 'r')
        # Creating an object Point to iterate over the board matrix
        p = Points(0, 0)
        for i in range(9):
            l = f.readline()
            for c in l:
                if c != '\n':
                    # if p is an empty square, sets the possible numbers to all possible {1,2,3,4,5,6,7,8,9}
                    if c == '0':
                        self.board[p.ij] = {1, 2, 3, 4, 5, 6, 7, 8, 9}
                    # Otherwise inputs the number to the board as a set containing only the integer
                    else:
                        self.board[p.ij] = {int(c)}
                    # Iterates p to the next box on the board
                    p.iterate()
        f.close()

    def import_board_list(self, file):
        p = Points(0, 0)
        for i in range(9):
            l = file.readline()
            for c in l:
                if c != '\n':
                    # if p is an empty square, sets the possible numbers to all possible {1,2,3,4,5,6,7,8,9}
                    if c == '0':
                        self.board[p.ij] = {1, 2, 3, 4, 5, 6, 7, 8, 9}
                    # Otherwise inputs the number to the board as a set containing only the integer
                    else:
                        self.board[p.ij] = {int(c)}
                    # Iterates p to the next box on the board
                    p.iterate()

    def print(self):
        p = Points(0, 0)
        for i in range(9):
            if i % 3 == 0:
                print('')
            for j in range(9):
                if j % 3 == 0:
                    print('  ', end='')
                if self.n_possible(p) == 1:
                    print(next(iter(self.board[p.ij])), end=' ')
                else:
                    print('_', end=' ')
                p.iterate()
            print('\n', end='')
        print('')

    def print_prep(self):
        c = copy.deepcopy(self)
        for i in range(9):
            for j in range(9):
                if len(c.board[i, j]) == 1:
                    c.board[i, j] = str(next(iter(c.board[i, j])))
                else:
                    c.board[i, j] = ' '
        return c.board

    def n_possible(self, point: Points):
        """ Return the amount of possible numbers in a box. """
        return len(self.board[point.ij])

    def fill_all_obvious(self):
        """ Checks the board to find obvious simplifications"""
        # Iterating through each case in the board
        p = Points(0, 0)
        change_occurred = True
        while change_occurred:
            change_occurred = False
            for a in range(9 ** 2):
                # If there are multiple possible numbers, then
                # iterating through box, line and column to check for possibilities to be removed
                if self.n_possible(p) > 1:
                    q = Points(p.i, p.j)

                    # Iterating through box to remove possibilities
                    q.box_iter()
                    for b in range(8):
                        if self.n_possible(q) == 1:
                            if self.board[p.ij] & self.board[q.ij]:
                                change_occurred = True
                            self.board[p.ij] -= self.board[q.ij]
                        q.box_iter()

                    # Iterating through line to remove possibilities
                    q.line_iter()
                    for l in range(8):
                        if self.n_possible(q) == 1:
                            if self.board[p.ij] & self.board[q.ij]:
                                change_occurred = True
                            self.board[p.ij] -= self.board[q.ij]
                        q.line_iter()

                    # Iterating through column to remove possibilities
                    q.col_iter()
                    for c in range(8):
                        if self.n_possible(q) == 1:
                            if self.board[p.ij] & self.board[q.ij]:
                                change_occurred = True
                            self.board[p.ij] -= self.board[q.ij]
                        q.col_iter()
                p.iterate()

    def is_solved(self):
        p = Points(0, 0)
        solved = True
        for i in range(9 ** 2):
            if self.n_possible(p) > 1:
                self.solved = False
                solved = False
            elif self.n_possible(p) == 0:
                self.solvable = False
            p.iterate()
        if solved:
            self.solved = True
        return solved
    
    def solve(self):
        self.fill_all_obvious()

        


def import_boards(file_name):
    # Open file containing the sudoku boards
    f = open(file_name, 'r')
    all_boards = []
    for i in range(50):
        line = f.readline()
        if line[0].isalpha():
            b = SudokuBoard()
            b.import_board_list(f)
            all_boards.append(b)
    f.close()
    return all_boards


def solve(b: SudokuBoard):
    # Fills all obvious cases
    b.fill_all_obvious()

    if b.is_solved() or not b.solvable:
        return b
    # General case: Find first undetermined case and suppose that the first possibility is true.
    # Then solve for the rest of the board. If the guess was wrong, repeat for next possibility.
    else:

        # Iterating through board to find the first undetermined case
        p = Points(0, 0)
        while b.n_possible(p) == 1:
            p.iterate()

        # Looping through all the possibilities
        possible = b.board[p.ij]
        for n in possible:

            # Making a copy of the board and replacing the undetermined case with a guess
            c = copy.deepcopy(b)
            c.board[p.ij] = {n}

            # Fills all obvious cases
            b.fill_all_obvious()

            # Trying to solve for the rest of the board. Returning True if the right number is found.
            s = solve(c)
            if solve(s).solved and s.solvable:
                return s
        return b


def solve_all_possible(s: SudokuBoard):
    global solutions
    solutions = []

    def solver(b: SudokuBoard):

        # Fill all obvious cases
        b.fill_all_obvious()

        if b.is_solved():
            b.print()
            solutions.append(b)
            return
        elif not b.solvable:
            return
        else:

            # Finding the first open case
            p = Points(0,0)
            while b.n_possible(p) == 1:
                p.iterate()

            possible = b.board[p.ij]
            for n in possible:

                # Making a copy of the board and replacing the open case with a guess
                c = copy.deepcopy(b)
                c.board[p.ij] = {n}

                solver(c)

    solver(s)
    return solutions





