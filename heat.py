# This script makes a simulation based on the explicit Euler method for the 2 heat equation
# Based on this readme: https://github.com/tommaso-ferracci/Heat_Equation_2D
import numpy as np

# This function is used to calculate the next state of the board
def next_state(board, dt=1, dx=1, dy=1, alpha=1, beta=1):
    # Create a new board to store the next state
    new_board = [[0 for _ in range(len(board[0]))] for _ in range(len(board))]
    # Iterate over the board
    for i in range(len(board)):
        for j in range(len(board[0])):
            # Have care of the boundaries
            if i == 0 or i == len(board) - 1 or j == 0 or j == len(board[0]) - 1:
                new_board[i][j] = board[i][j]
                continue
            # Calculate the next state of the board
            new_board[i][j] = board[i][j] + alpha * dt * (board[i-1][j] - 2 * board[i][j] + board[i+1][j]) / (dx ** 2) + beta * dt * (board[i][j-1] - 2 * board[i][j] + board[i][j+1]) / (dy ** 2)
    return np.array(new_board)

# This function is used to simulate the heat equation
def simulate_heat_equation(board, dt=1, dx=1, dy=1, alpha=1, beta=1, n_steps=10):
    board_sim = [board]
    # Iterate over the number of steps
    for _ in range(n_steps):
        # Calculate the next state of the board
        new_board = next_state(board_sim[-1], dt, dx, dy, alpha, beta)
        board_sim.append(new_board)
    return board_sim

# This function is used to compute a new board based on the ponderation of all the future boards
def score_ponderation(board_sim, n_steps=10, coef=0.1):
    # Create a new board to store the score
    score_board = np.zeros_like(board_sim[0])
    # Iterate over the boards
    for board in board_sim:
        # Update the score board
        score_board += board * (coef ** n_steps)
    return score_board

# This function is used to get the next move based on the score board
def get_sim_based_score(board, dt=1, dx=1, dy=1, alpha=1, beta=1, n_steps=10, coef=0.1):
    # Simulate the heat equation
    board_sim = simulate_heat_equation(board, dt, dx, dy, alpha, beta, n_steps)
    # Compute the score board
    score_board = score_ponderation(board_sim, n_steps, coef)
    # # Get the next move based on the score board
    # next_move = np.argmax(score_board)
    return score_board