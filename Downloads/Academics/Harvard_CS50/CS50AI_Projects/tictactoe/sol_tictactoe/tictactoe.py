"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Count the number of X's and O's on the board.
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)

    # If x_count is greater than o_count, it's O's turn.
    # Otherwise, it's X's turn (this covers the initial state where counts are equal).
    if x_count > o_count:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    # Iterate over each cell of the board.
    for i in range(3):
        for j in range(3):
            # If a cell is empty, it's a possible move.
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Unpack the row and column from the action tuple.
    i, j = action

    # Check if the action is valid. If not, raise an exception.
    if i < 0 or i >= 3 or j < 0 or j >= 3 or board[i][j] is not EMPTY:
        raise ValueError("Invalid action")

    # Create a deep copy of the board to avoid modifying the original.
    new_board = copy.deepcopy(board)

    # Determine whose turn it is and place their mark on the new board.
    current_player = player(board)
    new_board[i][j] = current_player

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows for a winner
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not EMPTY:
            return board[i][0]

    # Check columns for a winner
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] and board[0][j] is not EMPTY:
            return board[0][j]

    # Check diagonals for a winner
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not EMPTY:
        return board[0][2]

    # If no winner is found
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # The game is over if there is a winner or if the board is full.
    if winner(board) is not None or not actions(board):
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won, -1 if O has won, 0 otherwise.
    """
    win_player = winner(board)
    if win_player == X:
        return 1
    elif win_player == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # If the board is a terminal board, there are no moves to make.
    if terminal(board):
        return None

    current_player = player(board)

    if current_player == X:
        # X is the maximizer. We want to find the action that leads to the highest score.
        best_score = -math.inf
        best_action = None
        for action in actions(board):
            # The score for an action is the score of the resulting state for the opponent.
            score = min_value(result(board, action))
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
    else:  # current_player == O
        # O is the minimizer. We want to find the action that leads to the lowest score.
        best_score = math.inf
        best_action = None
        for action in actions(board):
            # The score for an action is the score of the resulting state for the opponent.
            score = max_value(result(board, action))
            if score < best_score:
                best_score = score
                best_action = action
        return best_action

# Helper function for the maximizer (X)
def max_value(board):
    """
    Returns the maximum utility that the maximizer can get from the current board.
    """
    if terminal(board):
        return utility(board)

    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v

# Helper function for the minimizer (O)
def min_value(board):
    """
    Returns the minimum utility that the minimizer can get from the current board.
    """
    if terminal(board):
        return utility(board)

    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v
