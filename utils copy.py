# This file has relevant functions to evaluate the game state and make decisions based on it.
import numpy as np

# This function extracts the tile locations of a snake's body from the game state
def get_snake_body(snake):
    return snake["body"]

# This function gets the body of all snakes in the game state
def get_all_snake_bodies(board):
    return [get_snake_body(snake) for snake in board["snakes"]]

# This function gets the food locations from the game state
def get_food_locations(board):
    return board["food"]

# This function gets the hazards from the game state
def get_hazards(board):
    return board["hazards"]

# This function scores the accessibility of a tile based on the number of accessible tiles around it
# To assess the accessibility of a tile, it should not be occupied by a snake or a hazard
def score_tile_accessibility(board, tile, snake_bodies=None, hazards=None, width=None, height=None):
    x, y = tile
    score = 0
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        new_x, new_y = x + dx, y + dy
        if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
            continue
        if (new_x, new_y) in hazards:
            continue
        if any((new_x, new_y) in body for body in snake_bodies):
            continue
        score += 1
    return -np.sqrt(1-score/4)

# This function scores all the tiles in the board based on their accessibility
def score_board_accessibility(board):
    width = board["width"]
    height = board["height"]
    hazards = get_hazards(board)
    hazards = [(hazard["x"], hazard["y"]) for hazard in hazards]
    snake_bodies = get_all_snake_bodies(board)
    # For the snake bodies, delete the last element of each snake body, because the last element is the tail (If the body is larger than 1)
    snake_bodies = [body[:-1] if len(body) > 1 else body for body in snake_bodies]
    snake_bodies = [[(tile["x"], tile["y"]) for tile in body] for body in snake_bodies]
    board_score = np.array([[score_tile_accessibility(board, (x, y), snake_bodies, hazards, width, height) for x in range(width)] for y in range(height)])
    # # Now normalize the board score from -1 to 0
    # board_score = (board_score - np.min(board_score)) / (np.max(board_score) - np.min(board_score)) - 1
    # Center the max score to 0
    board_score = board_score - np.max(board_score)
    return board_score

# This function scores the distance of a tile from the head of a snake
# The score is a power law function of the distance
# The distance is calculated as the Manhattan distance
def score_tile_distance_from_head(snake, tile, alpha=2):
    head = snake["head"]
    distance = np.abs(head["x"] - tile["x"]) + np.abs(head["y"] - tile["y"])
    return -1/np.power(alpha, distance)

# This function scores the distance of all tiles from the head of a snake
def score_board_distance_from_head(grid, snake, alpha=2):
    return np.array([[score_tile_distance_from_head(snake, tile, alpha) for tile in row] for row in grid])

# This function scores the distance of a tile from the body of a snake
# The score is a power law function of the distance
# The distance is calculated as the Manhattan distance
def score_board_distance_from_body(grid, snake, height, width, alpha=8):
    body = snake["body"]
    body_score = np.zeros((height, width))
    for tile in body:
        x, y = tile["x"], tile["y"]
        body_score += np.array([[score_tile_distance_from_head(snake, tile, alpha) for tile in row] for row in grid])
    return body_score

# This function scores the distance of a tile from the food
# The score is an exponential function of the distance
def score_tile_distance_from_food(tile, food, beta=1):
    distance = np.abs(food["x"] - tile["x"]) + np.abs(food["y"] - tile["y"])
    return np.exp(-distance*beta)

# This function scores the distance of all tiles from the food
def score_board_distance_from_food(grid, food, beta=None, delta=1, gamma=1, health=100):
    if beta is None:
        beta = 1/(100-health+delta)**gamma
    return np.array([[score_tile_distance_from_food(tile, food, beta) for tile in row] for row in grid])

# This function scores the board by combining the accessibility and distance from head and food scores
def score_board(board, snakes, foods, alpha=2, beta=None, delta=1, gamma=1, health=100):
    grid = [[{"x": x, "y": y} for x in range(board["width"])] for y in range(board["height"])]
    accessibility_score = score_board_accessibility(board)
    # print(f"ACCESSIBILITY SCORE: {accessibility_score}")
    head_scores = np.zeros((board["height"], board["width"]))
    for id, snake in enumerate(snakes):
        if id == 0:
            continue
        head_scores += score_board_distance_from_head(grid, snake, alpha)
    if len(snakes) > 1:
        head_scores /= len(snakes)-1
    body_scores = np.zeros((board["height"], board["width"]))
    for id, snake in enumerate(snakes):
        body_scores += score_board_distance_from_body(grid, snake, board["height"], board["width"], alpha+len(snake["body"]))
    body_scores /= len(snakes)
    # print(f"HEAD SCORES: {head_scores}")
    food_scores = np.zeros((board["height"], board["width"]))
    for food in foods:
        food_scores += score_board_distance_from_food(grid, food, beta, delta, gamma, health)
    food_scores /= len(foods)
    # print(f"FOOD SCORES: {food_scores}")
    return (accessibility_score + head_scores + body_scores + food_scores) / 4

# Now parse the board score to avoid impossible moves
def parse_board_score(board_score, snakes, foods):
    for snake in snakes:
        # Ignore the tails in the snake bodies
        # print(f"SNAKE BODY: {snake['body']}")
        if len(snake["body"]) > 1:
            snake_body = snake["body"][:-1]
        else:
            snake_body = snake["body"]
        for body in snake_body:
            board_score[body["y"], body["x"]] = -np.inf
    # for food in foods:
    #     board_score[food["y"], food["x"]] = 1
    return board_score

# This function gets the best move from the board score
def get_best_move(board_score, head):
    x, y = head["x"], head["y"]
    best_move = None
    move_tiers = []
    best_score = -np.inf
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        new_x, new_y = x + dx, y + dy
        if not (new_x < 0 or new_x >= board_score.shape[1] or new_y < 0 or new_y >= board_score.shape[0]):
            if board_score[new_y, new_x] > -np.inf:
                move_tiers.append((dx, dy, board_score[new_y, new_x]))
            if board_score[new_y, new_x] > best_score:
                best_score = board_score[new_y, new_x]
                best_move = (dx, dy)
    return best_move, list(sorted(move_tiers, key=lambda x: x[2], reverse=True))

# This function runs flood fill to count the number of accessible tiles from a given starting position
def count_accessible_tiles(board, start, snake_bodies, hazards, width, height):
    x, y = start
    accessible_tiles = 1
    # Initialize the visited array
    visited = np.zeros((height, width))
    visited[y, x] = 1
    # Initialize the stack of tiles to visit
    stack = [(x, y)]
    while len(stack) > 0:
        # Pop the last tile from the stack
        x, y = stack.pop()
        # Visit the adjacent tiles
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x, new_y = x + dx, y + dy
            # Check if the new tile is accessible
            if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
                continue
            # Check if the new tile is a hazard
            if (new_x, new_y) in hazards:
                continue
            if any((new_x, new_y) in body for body in snake_bodies):
                continue
            #
            # If the tile is legitimate and not visited, add it to the stack
            if visited[new_y, new_x] == 0:
                visited[new_y, new_x] = 1
                accessible_tiles += 1
                stack.append((new_x, new_y))
    return accessible_tiles


# This function runs over the accessible tiles and scores them based on the number of accessible tiles
def score_accessible_tiles(board_score, start, snake_bodies, hazards, width, height):
    x, y = start
    # Initialize the visited array
    visited = np.zeros((height, width))
    visited[y, x] = 1
    print(board_score)
    score_sum = board_score[y][x]
    # Initialize the stack of tiles to visit
    stack = [(x, y)]
    while len(stack) > 0:
        # Pop the last tile from the stack
        x, y = stack.pop()
        # Visit the adjacent tiles
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x, new_y = x + dx, y + dy
            # Check if the new tile is accessible
            if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
                continue
            # Check if the new tile is a hazard
            if (new_x, new_y) in hazards:
                continue
            if any((new_x, new_y) in body for body in snake_bodies):
                continue
            #
            # If the tile is legitimate and not visited, add it to the stack
            if visited[new_y, new_x] == 0:
                visited[new_y, new_x] = 1
                score_sum += board_score[new_y, new_x]
                stack.append((new_x, new_y))
    return score_sum

# This function gets the next move based on the game state
def get_next_move(game_state):
    board = game_state["board"]
    snakes = board["snakes"]
    foods = board["food"]
    my_snake = game_state["you"]
    head = my_snake["head"]
    tail = my_snake["body"][-1]
    #
    health = my_snake["health"]
    delta = 1
    gamma = 0.5
    beta = 1/(100-health+delta)**gamma
    #
    board_score = score_board(board, snakes, foods, beta=beta)
    board_score = parse_board_score(board_score, snakes, foods)
    best_move, move_tiers = get_best_move(board_score, head)
    
    print(f"MOVE TIERS: {move_tiers}")
    if len(move_tiers) > 1:
        hazards = get_hazards(board)
        hazards = [(hazard["x"], hazard["y"]) for hazard in hazards]
        snake_bodies = get_all_snake_bodies(board)
        # For the snake bodies, delete the last element of each snake body, because the last element is the tail (If the body is larger than 1)
        snake_bodies = [body[:-1] if len(body) > 1 else body for body in snake_bodies]
        snake_bodies = [[(tile["x"], tile["y"]) for tile in body] for body in snake_bodies]
        #
        # If the two best moves are similar (Percentual difference less than 10%), choose the best move based on the number of accessible tiles
        if abs((move_tiers[0][2] - move_tiers[1][2]) / move_tiers[0][2]) < 0.1:
            # For every move, count the number of accessible tiles from the new head position
            for i, (dx, dy, score) in enumerate(move_tiers):
                new_x, new_y = head["x"] + dx, head["y"] + dy
                accessible_tiles = count_accessible_tiles(board, (new_x, new_y), snake_bodies, hazards, board["width"], board["height"])
                # Now ponder the moves based on the number of accessible tiles and the score
                # The ponderation considers that accessible tiles has a range from [0, width*height] and the score has a range from [-1, 1]
                # Hence, we change the score range to [0, 2] and ponder the score by the number of accessible tiles
                pondered_score = (score+1) * (accessible_tiles / (board["width"] * board["height"]))
                # pondered_score = accessible_tiles
                move_tiers[i] = (dx, dy, score, accessible_tiles, pondered_score)
            move_tiers = sorted(move_tiers, key=lambda x: x[4], reverse=True)
            print(f"UPDATE MOVE TIERS: {move_tiers}")
            print(f"DIFFERENCE: {abs((move_tiers[0][4] - move_tiers[1][4]) / move_tiers[0][4])}")
            best_move = (move_tiers[0][0], move_tiers[0][1])
            if len(my_snake["body"]) > board["width"]*board["height"]/2:
                # If the two best moves are similar (Percentual difference less than 25%), follow your tail :p
                if (abs((move_tiers[0][4] - move_tiers[1][4]) / move_tiers[0][4]) < 0.1):
                    # See which move is closer to the tail
                    tail_distance = np.inf
                    for i, (dx, dy, score, accessible_tiles, pondered_score) in enumerate(move_tiers):
                        new_x, new_y = head["x"] + dx, head["y"] + dy
                        distance = np.abs(tail["x"] - new_x) + np.abs(tail["y"] - new_y)
                        if distance < tail_distance:
                            best_move = (dx, dy)
                            tail_distance = distance
                # Sum the scores of the two enclosed areas (The potential area accessible by the two best moves)
                elif abs((move_tiers[0][4] - move_tiers[1][4]) / move_tiers[0][4]) < 0.25:
                    # For every move, count the number of accessible tiles from the new head position
                    for i, (dx, dy, score, accessible_tiles, pondered_score) in enumerate(move_tiers):
                        new_x, new_y = head["x"] + dx, head["y"] + dy
                        score_sum = score_accessible_tiles(board_score, (new_x, new_y), snake_bodies, hazards, board["width"], board["height"])
                        move_tiers[i] = (dx, dy, score, accessible_tiles, pondered_score, score_sum)
                    move_tiers = list(sorted(move_tiers, key=lambda x: x[5], reverse=True))
                    print(f"UPDATE MOVE TIERS: {move_tiers}")
                    best_move = (move_tiers[0][0], move_tiers[0][1])
        #
        # If the two best moves are opposite, choose the best move based on the number of accessible tiles
        elif (move_tiers[0][0] == -move_tiers[1][0]) and (move_tiers[0][1] == -move_tiers[1][1]):
            # For every move, count the number of accessible tiles from the new head position
            for i, (dx, dy, score) in enumerate(move_tiers):
                new_x, new_y = head["x"] + dx, head["y"] + dy
                accessible_tiles = count_accessible_tiles(board, (new_x, new_y), snake_bodies, hazards, board["width"], board["height"])
                # Now ponder the moves based on the number of accessible tiles and the score
                # The ponderation considers that accessible tiles has a range from [0, width*height] and the score has a range from [-1, 1]
                # Hence, we change the score range to [0, 2] and ponder the score by the number of accessible tiles
                pondered_score = (score+1) * (accessible_tiles / (board["width"] * board["height"]))
                pondered_score = accessible_tiles
                move_tiers[i] = (dx, dy, score, accessible_tiles, pondered_score)
            move_tiers = sorted(move_tiers, key=lambda x: x[4], reverse=True)
            print(f"UPDATE MOVE TIERS: {move_tiers}")
            best_move = (move_tiers[0][0], move_tiers[0][1])


    # Sort the moves by pondering
    if best_move == (1, 0):
        return "right", board_score
    elif best_move == (-1, 0):
        return "left", board_score
    elif best_move == (0, -1):
        return "down", board_score
    elif best_move == (0, 1):
        return "up", board_score
    else:
        return "down", board_score