# This file has relevant functions to evaluate the game state and make decisions based on it.
import numpy as np
from heat import get_sim_based_score

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
    # Now normalize the sum of the board score to 1 and center the max score to 0
    # board_score = (board_score - np.min(board_score)) / (np.max(board_score) - np.min(board_score))
    board_score = board_score - np.min(board_score)
    board_score = board_score/np.sum(board_score)
    board_score = board_score - np.max(board_score)
    # # Send the board score to positive values
    # board_score = board_score - np.min(board_score)
    # # Normalize the board score
    # board_score = board_score / np.sum(board_score)
    # # Center the max score to 0
    # board_score = board_score - np.max(board_score)
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
    distance_score = np.array([[score_tile_distance_from_head(snake, tile, alpha) for tile in row] for row in grid])
    # # Normalize the distance score and center it to 0
    # distance_score = distance_score / np.sum(distance_score) - 1
    # distance_score = (distance_score - np.min(distance_score)) / (np.max(distance_score) - np.min(distance_score))
    distance_score = distance_score - np.min(distance_score)
    distance_score = distance_score / np.sum(distance_score)
    distance_score = distance_score - np.max(distance_score)
    return distance_score

# This function scores the distance of a tile from the body of a snake
# The score is a power law function of the distance
# The distance is calculated as the Manhattan distance
def score_board_distance_from_body(grid, snake, height, width, alpha=8):
    body = snake["body"]
    body_score = np.zeros((height, width))
    for id, tile in enumerate(body):
        if id == 0:
            continue
        x, y = tile["x"], tile["y"]
        # Decrease the score of the tile by the distance from the head
        distance = id/len(body)
        #np.abs(snake["head"]["x"] - x) + np.abs(snake["head"]["y"] - y)
        alpha_tmp = alpha**distance
        body_score += np.array([[score_tile_distance_from_head(snake, tile, alpha_tmp) for tile in row] for row in grid])
    # Normalize the body score and center it to 0
    # body_score = body_score / np.sum(body_score) - 1
    # body_score = (body_score - np.min(body_score)) / (np.max(body_score) - np.min(body_score))
    body_score = body_score - np.min(body_score)
    body_score = body_score / np.sum(body_score)
    body_score = body_score - np.max(body_score)
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
    food_score = np.array([[score_tile_distance_from_food(tile, food, beta) for tile in row] for row in grid])
    # Normalize the food score and keep it in the range [0, 1]
    # food_score = food_score / np.sum(food_score)
    # food_score = (food_score - np.min(food_score)) / (np.max(food_score) - np.min(food_score))
    food_score = food_score - np.min(food_score)
    food_score = food_score / np.sum(food_score)
    return food_score

# This function scores the board by combining the accessibility and distance from head and food scores
def score_board(board, snakes, foods, my_snake, alpha=None, beta=None, delta=1, gamma=1, health=100):
    grid = [[{"x": x, "y": y} for x in range(board["width"])] for y in range(board["height"])]
    accessibility_score = score_board_accessibility(board)
    # print(f"ACCESSIBILITY SCORE: {accessibility_score}")
    head_scores = np.zeros((board["height"], board["width"]))
    for snake in snakes:
        id = snake["id"]
        if id == my_snake["id"]:
            continue
        if alpha is None:
            alpha_tmp = 1+(len(my_snake["body"])-1)/len(snake["body"])
        else:
            alpha_tmp = alpha
        head_scores += score_board_distance_from_head(grid, snake, alpha=alpha_tmp)
    if len(snakes) > 1:
        head_scores /= len(snakes)-1
    body_scores = np.zeros((board["height"], board["width"]))
    for id, snake in enumerate(snakes):
        if alpha is None:
            alpha_tmp = 1+(my_snake["health"]*len(my_snake["body"]))/(len(snake["body"])*snake["health"])
        else:
            alpha_tmp = alpha
        body_scores += score_board_distance_from_body(grid, snake, board["height"], board["width"], alpha=alpha_tmp)
    body_scores /= len(snakes)
    # print(f"HEAD SCORES: {head_scores}")
    food_scores = np.zeros((board["height"], board["width"]))
    for food in foods:
        food_scores += score_board_distance_from_food(grid, food, beta, delta, gamma, health)
    food_scores /= len(foods)
    # print(f"FOOD SCORES: {food_scores}")
    return (accessibility_score + head_scores + body_scores + food_scores) / 4

# Now parse the board score to avoid impossible moves
def parse_board_score(board_score, snakes, foods, my_snake=None):
    for snake in snakes:
        # Ignore the tails in the snake bodies
        # print(f"SNAKE BODY: {snake['body']}")
        if len(snake["body"]) > 1:
            snake_body = snake["body"][:-1]
        else:
            snake_body = snake["body"]
        for body in snake_body:
            board_score[body["y"], body["x"]] = -np.inf
    for food in foods:
        if my_snake is not None:
            board_score[food["y"], food["x"]] += (100-my_snake["health"])/100
        # board_score[food["y"], food["x"]] = 1
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

# This function finds the local maximum from a given position by doing an iterative search
def find_local_maximum(board_score, start):
    x, y = start
    new_x, new_y = x, y
    max_score = board_score[y, x]
    sum_to_max = max_score
    distance_to_max = 1
    while True:
        found = False
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_x, next_y = new_x + dx, new_y + dy
            if next_x < 0 or next_x >= board_score.shape[1] or next_y < 0 or next_y >= board_score.shape[0]:
                continue
            if board_score[next_y, next_x] > max_score:
                max_score = board_score[next_y, next_x]
                new_x, new_y = next_x, next_y
                found = True
                distance_to_max += 1
                sum_to_max += max_score
        if not found:
            break
    return {"max": (new_x, new_y), "distance": distance_to_max, "sum": sum_to_max}

# This function gets the next move based on the game state
def get_next_move(game_state):
    board = game_state["board"]
    snakes = board["snakes"]
    foods = board["food"]
    my_snake = game_state["you"]
    head = my_snake["head"]
    board_score = score_board(board, snakes, foods, my_snake, alpha=None, beta=None, delta=1, gamma=2, health=my_snake["health"])
    board_score = get_sim_based_score(board_score, dt=1/15,
                                    #   dx=1, dy=1,
                                      dx=1/len(board_score[0]), dy=1/len(board_score),
                                      alpha=0.01, beta=0.01, n_steps=15, coef=0.35)
    board_score = parse_board_score(board_score, snakes, foods, my_snake=my_snake)
    best_move, move_tiers = get_best_move(board_score, head)

    print(f"MOVE TIERS: {move_tiers}")
    if len(move_tiers) > 1:
        hazards = get_hazards(board)
        hazards = [(hazard["x"], hazard["y"]) for hazard in hazards]
        snake_bodies = get_all_snake_bodies(board)
        # For the snake bodies, delete the last element of each snake body, because the last element is the tail (If the body is larger than 1)
        snake_bodies = [body[:len(body)-1] if len(body) > 1 else body for body in snake_bodies]
        snake_bodies = [[(tile["x"], tile["y"]) for tile in body] for body in snake_bodies]
        #
        # If the two best moves are opposite, choose the best move based on the number of accessible tiles
        if (move_tiers[0][0] == -move_tiers[1][0]) and (move_tiers[0][1] == -move_tiers[1][1]):
            # For every move, count the number of accessible tiles from the new head position
            min_score = move_tiers[-1][2]
            for i, (dx, dy, score) in enumerate(move_tiers):
                new_x, new_y = head["x"] + dx, head["y"] + dy
                accessible_tiles = count_accessible_tiles(board, (new_x, new_y), snake_bodies, hazards, board["width"], board["height"])
                # Now ponder the moves based on the number of accessible tiles and the score
                # The ponderation considers that accessible tiles has a range from [0, width*height] and the score has a range from [-1, 1]
                # Hence, we change the score range to [0, 2] and ponder the score by the number of accessible tiles
                pondered_score = (score-min_score+1) * (accessible_tiles / (board["width"] * board["height"]))
                # pondered_score = accessible_tiles
                move_tiers[i] = (dx, dy, score, accessible_tiles, pondered_score)
            move_tiers = sorted(move_tiers, key=lambda x: x[4], reverse=True)
            print(f"UPDATE MOVE TIERS: {move_tiers}")
            best_move = (move_tiers[0][0], move_tiers[0][1])
        # If the two best moves are similar (Percentual difference less than 10%), choose the best move based on the number of accessible tiles
        elif abs((move_tiers[0][2] - move_tiers[1][2]) / move_tiers[0][2]) < 0.1:
            # For every move, count the number of accessible tiles from the new head position
            min_score = move_tiers[-1][2]
            for i, (dx, dy, score) in enumerate(move_tiers):
                new_x, new_y = head["x"] + dx, head["y"] + dy
                accessible_tiles = count_accessible_tiles(board, (new_x, new_y), snake_bodies, hazards, board["width"], board["height"])
                # Now ponder the moves based on the number of accessible tiles and the score
                # The ponderation considers that accessible tiles has a range from [0, width*height] and the score has a range from [-1, 1]
                # Hence, we change the score range to [0, 2] and ponder the score by the number of accessible tiles
                pondered_score = (score-min_score+1) * (accessible_tiles / (board["width"] * board["height"]))
                # pondered_score = accessible_tiles
                move_tiers[i] = (dx, dy, score, accessible_tiles, pondered_score)
            move_tiers = sorted(move_tiers, key=lambda x: x[4], reverse=True)
            print(f"UPDATE MOVE TIERS: {move_tiers}")
            best_move = (move_tiers[0][0], move_tiers[0][1])
        # If the best move has little space to move, choose the second best move
        elif move_tiers[0][2] < -0.5:
            # Check which move has the most accessible tiles
            min_score = move_tiers[-1][2]
            for i, (dx, dy, score) in enumerate(move_tiers):
                new_x, new_y = head["x"] + dx, head["y"] + dy
                accessible_tiles = count_accessible_tiles(board, (new_x, new_y), snake_bodies, hazards, board["width"], board["height"])
                # Now ponder the moves based on the number of accessible tiles and the score
                # The ponderation considers that accessible tiles has a range from [0, width*height] and the score has a range from [-1, 1]
                # Hence, we change the score range to [0, 2] and ponder the score by the number of accessible tiles
                pondered_score = (score-min_score+1) * (accessible_tiles / (board["width"] * board["height"]))
                # pondered_score = accessible_tiles
                move_tiers[i] = (dx, dy, score, accessible_tiles, pondered_score)
            move_tiers = sorted(move_tiers, key=lambda x: x[4], reverse=True)
            print(f"UPDATE MOVE TIERS: {move_tiers}")
            best_move = (move_tiers[0][0], move_tiers[0][1])
        #
        # Now, if all the moves keep being the similar, check based on closeness to the local maximum of each move
        if len(move_tiers[0]) > 3:
            if abs((move_tiers[0][4] - move_tiers[1][4]) / move_tiers[0][4]) < 0.1:
                # Find the local maximum for each move and choose the one that is closer to the local maximum
                for i, (dx, dy, score, accessible_tiles, pondered_score) in enumerate(move_tiers):
                    new_x, new_y = head["x"] + dx, head["y"] + dy
                    maximum_value_dict = find_local_maximum(board_score, (new_x, new_y))
                    sum_to_local_maximum = maximum_value_dict["sum"]
                    distance_to_local_maximum = maximum_value_dict["distance"]
                    maximum_value = board_score[local_maximum[1], local_maximum[0]]
                    new_score = ((sum_to_local_maximum+distance_to_local_maximum)*0.5/distance_to_local_maximum) # Center the scores to 0 and normalize to 1 and ponder by the distance to the local maximum
                    new_score = new_score * pondered_score
                    move_tiers[i] = (dx, dy, score, accessible_tiles, pondered_score, maximum_value, distance_to_local_maximum, new_score)
                move_tiers = sorted(move_tiers, key=lambda x: x[7], reverse=True)
                print(f"UPDATE MOVE TIERS: {move_tiers}")
                best_move = (move_tiers[0][0], move_tiers[0][1])
        #
        # See if my snake is going to a trap with other snake (an apparent local maximum but that has no way out in the future)
        # First, find the local maximum location for the best move
        new_x, new_y = head["x"] + best_move[0], head["y"] + best_move[1]
        local_maximum_dict = find_local_maximum(board_score, (new_x, new_y))
        local_maximum = local_maximum_dict["max"]
        # Now check if the immediate radius (distance of sqrt(length of my snake body)) around the local maximum is a trap
        # That is, has at least a number of accessible tiles equal to the length of my snake body
        radius = int(np.sqrt(len(my_snake["body"])))
        free_tiles = 0
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                x, y = local_maximum[0] + dx, local_maximum[1] + dy
                if x < 0 or x >= board_score.shape[1] or y < 0 or y >= board_score.shape[0]:
                    continue
                if board_score[y, x] > -np.inf:
                    free_tiles += 1
        if free_tiles < radius//2:
            # If the best move is a trap, choose the second best move
            print(f"TRAP DETECTED: {free_tiles}")
            move_tiers.pop(0)
            best_move = (move_tiers[0][0], move_tiers[0][1])
        #
        # If the best move is a direct crash with a snake with more length, choose the second best move
        for snake in snakes:
            if snake["id"] == my_snake["id"]:
                continue
            if len(snake["body"]) == len(my_snake["body"]) and len(snakes) == 2:
                continue
            if len(snake["body"]) >= len(my_snake["body"]):
                if len(move_tiers) > 1:
                    # See the distance of every move from the snake head
                    for move in move_tiers:
                        destiny_tile = (head["x"] + move[0], head["y"] + move[1])
                        distance_with_snake = np.abs(snake["head"]["x"] - destiny_tile[0]) + np.abs(snake["head"]["y"] - destiny_tile[1])
                        move_tiers[move_tiers.index(move)] = (*move, distance_with_snake)
                    # See how many moves are at distance 1 from my head and the snake head
                    # If all the moves are at distance 1, choose the best move
                    # If at least one move is further than distance 1, choose the one with the highest score
                    move_tiers_at_distance_1 = [move for move in move_tiers if move[-1] == 1]
                    if len(move_tiers_at_distance_1) > 0:
                        # If all the moves are at distance 1, choose the best move
                        if len(move_tiers_at_distance_1) == len(move_tiers):
                            pass
                        #
                        #
                        # Get the best move different from the one that is at distance 1
                        else:
                            for move in move_tiers:
                                if move[-1] > 1:
                                    best_move = (move[0], move[1])
                                    print(f"CHANGE MOVE: {best_move}")
                                    break

                # while len(move_tiers) > 1:
                #     # See if the move is at distance 1 from my head and the snake head
                #     destiny_tile = (head["x"] + best_move[0], head["y"] + best_move[1])
                #     distance_with_snake = np.abs(snake["head"]["x"] - destiny_tile[0]) + np.abs(snake["head"]["y"] - destiny_tile[1])
                #     if distance_with_snake == 1:
                #         # Pop the best move and choose the second best move
                #         move_tiers.pop(0)
                #         best_move = (move_tiers[0][0], move_tiers[0][1])
                #         print(f"CHANGE MOVE TIERS: {move_tiers}")
                #     else:
                #         break


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