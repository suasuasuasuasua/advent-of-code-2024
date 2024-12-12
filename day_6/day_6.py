"""
--- Day 6: Guard Gallivant ---

The Historians use their fancy device again, this time to whisk you all away to
the North Pole prototype suit manufacturing lab... in the year 1518! It turns
out that having direct access to history is very convenient for a group of
historians.

You still have to be careful of time paradoxes, and so it will be important to
avoid anyone from 1518 while The Historians search for the Chief. Unfortunately,
a single guard is patrolling this part of the lab.

Maybe you can work out where the guard will go ahead of time so that The
Historians can search safely?

You start by making a map (your puzzle input) of the situation. For example:

....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#...

The map shows the current position of the guard with ^ (to indicate the guard is
currently facing up from the perspective of the map). Any obstructions - crates,
desks, alchemical reactors, etc. - are shown as #.

Lab guards in 1518 follow a very strict patrol protocol which involves
repeatedly following these steps:

- If there is something directly in front of you, turn right 90 degrees.
- Otherwise, take a step forward.

Following the above protocol, the guard moves up several times until she reaches
an obstacle (in this case, a pile of failed suit prototypes):

....#.....
....^....#
..........
..#.......
.......#..
..........
.#........
........#.
#.........
......#...

Because there is now an obstacle in front of the guard, she turns right before
continuing straight in her new facing direction:

....#.....
........>#
..........
..#.......
.......#..
..........
.#........
........#.
#.........
......#...

Reaching another obstacle (a spool of several very long polymers), she turns
right again and continues downward:

....#.....
.........#
..........
..#.......
.......#..
..........
.#......v.
........#.
#.........
......#...

This process continues for a while, but the guard eventually leaves the mapped
area (after walking past a tank of universal solvent):

....#.....
.........#
..........
..#.......
.......#..
..........
.#........
........#.
#.........
......#v..

By predicting the guard's route, you can determine which specific positions in
the lab will be in the patrol path. Including the guard's starting position, the
positions visited by the guard before leaving the area are marked with an X:

....#.....
....XXXXX#
....X...X.
..#.X...X.
..XXXXX#X.
..X.X.X.X.
.#XXXXXXX.
.XXXXXXX#.
#XXXXXXX..
......#X..

In this example, the guard will visit 41 distinct positions on your map.

Predict the path of the guard. How many distinct positions will the guard visit
before leaving the mapped area?
"""

from pathlib import Path

import numpy as np

input_file = Path("sample.txt")
input_file = Path("input.txt")

with open(input_file, "r") as file_in:
    og_board = [[c for c in row.strip()] for row in file_in.readlines()]

og_board = np.array(og_board)

move_rotations = {
    # Up
    "^": ">",
    # Right
    ">": "v",
    # Down
    "v": "<",
    # Left
    "<": "^",
}


def find_guard(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Ref:
    #     https://stackoverflow.com/a/27175491 -- sorry bad with numpy
    mask = np.asarray(np.isin(board, list(move_rotations.keys())))
    char = board[mask]
    locations = np.nonzero(mask)

    # There should only be a singular location for the guard!
    ys, xs = locations
    assert ys.size == 1 and xs.size == 1

    # https://stackoverflow.com/a/43641281
    # Safely convert 1, array to scalar
    return char.item(), (ys.item(), xs.item())


def find_next_pos(
    board: np.ndarray, direction: str, y: int, x: int
) -> tuple[np.ndarray, np.ndarray]:
    # Default values (but should be overriden no matter what!)
    next_x, next_y = -1, -1
    moves = {}

    height, width = board.shape

    # Find the obstacles on the board (2D matrix remember!)
    mask = np.isin(board, ["#"])

    match direction:
        case "^":
            locs = mask[:, x].nonzero()

            # Destructure the locations because it's a tuple for some reason
            (ys,) = locs
            # Only consider the obstacles that have an index less
            # # (idx=0)
            #
            #
            # ^ (idx=3)
            #
            # # (idx=5) (don't look here)
            ys = ys[ys <= y]

            # If we have any viable obstacles, set the next locations
            if ys.size:
                # The X (LR) is the same since we're in the same file still
                next_x = x
                # The Y (UD) will be one unit away from the closest obstacle
                next_y = ys[-1] + 1

                # Capture the moves (go backward in y-direction)
                moves = {(yi, x) for yi in range(y, next_y - 1, -1)}

            # If there are no obstacles, walk off the board!
            else:
                next_x = x
                next_y = -1

                # Capture the moves (go backward in y-direction)
                moves = {(yi, x) for yi in range(y, next_y, -1)}

        case "v":
            locs = mask[:, x].nonzero()

            # Destructure the locations because it's a tuple for some reason
            (ys,) = locs
            # Only consider the obstacles that have an index less
            # # (idx=0) (don't look here)
            #
            #
            # v (idx=3)
            #
            # # (idx=5)
            ys = ys[ys >= y]

            # If we have any viable obstacles, set the next locations
            if ys.size:
                # The X (LR) is the same since we're in the same file still
                next_x = x
                # The Y (UD) will be one unit away from the closest obstacle
                next_y = ys[0] - 1

                # Capture the moves (go forward in y-direction)
                moves = {(yi, x) for yi in range(y, next_y + 1, 1)}

            # If there are no obstacles, walk off the board!
            else:
                next_x = x
                next_y = height

                # Capture the moves (go forward in y-direction)
                moves = {(yi, x) for yi in range(y, next_y, 1)}

        case ">":
            locs = mask[y, :].nonzero()

            # Destructure the locations because it's a tuple for some reason
            (xs,) = locs
            # Only consider the obstacles that have an index less
            # (idx=0)   (idx=1)    (idx=2) (look here!)
            #    #         >          #
            xs = xs[xs >= x]

            # If we have any viable obstacles, set the next locations
            if xs.size:
                # The X (LR) will be one unit away from the closest
                next_x = xs[0] - 1
                # The Y (UD) is the same since we're in the same row still
                next_y = y

                # Capture the moves (go forward)
                moves = {(y, xi) for xi in range(x, next_x + 1, 1)}

            # If there are no obstacles, walk off the board!
            else:
                next_x = width
                next_y = y

                # Capture the moves (go forward)
                moves = {(y, xi) for xi in range(x, next_x, 1)}

        case "<":
            locs = mask[y, :].nonzero()

            # Destructure the locations because it's a tuple for some reason
            (xs,) = locs
            # Only consider the obstacles that have an index less
            # (idx=0) (look here!)  (idx=1)    (idx=2)
            #    #                     <          #
            xs = xs[xs <= x]

            # If we have any viable obstacles, set the next locations
            if xs.size:
                # The X (LR) will be one unit away from the closest
                next_x = xs[-1] + 1
                # The Y (UD) is the same since we're in the same row still
                next_y = y

                # Capture the moves (go forward)
                moves = {(y, xi) for xi in range(x, next_x - 1, -1)}

            # If there are no obstacles, walk off the board!
            else:
                next_x = -1
                next_y = y

                # Capture the moves (go forward)
                moves = {(y, xi) for xi in range(x, next_x, -1)}

        case _:
            print(f"Something has gone horribly wrong!")
            exit(-1)

    return next_y, next_x, moves


def out_of_bounds(board: np.ndarray, next_pos: tuple[int, int]) -> bool:  #
    height, width = board.shape
    next_y, next_x = next_pos

    return next_y >= height or next_y < 0 or next_x >= width or next_x < 0


def move(
    board: np.ndarray, direction: str, location: tuple[int, int]
) -> tuple[np.ndarray, set[tuple[int, int]], tuple[int, int]]:
    # Find the span between the current position and the next obstacle ("#")
    y, x = location
    next_y, next_x, moves = find_next_pos(board, direction, y, x)

    # Erase the previous position
    board[y, x] = "."
    next_pos = next_y, next_x

    # Move the guard and rotate 90 degrees
    board[next_y, next_x] = move_rotations[direction]

    return board, moves, next_pos


# Part 1: Find all distinct location for the guard
distinct_moves = set()
board = og_board.copy()
while True:
    direction, location = find_guard(board)
    board, moves, next_pos = move(board, direction, location)

    distinct_moves |= moves

    if out_of_bounds(board, next_pos):
        break

p1 = len(distinct_moves)

print(f"Part 1: {p1}")

# Part 2 (got help from Reddit xd)
# My initial thought was to add one obstacle to each and every square (brute
# force)
# I also thought about trying to find "bounding boxes" (idk cause shapes are
# bigger)
#
# Many people just ended up doing the brute which I guess is fine...
#
# I think I basically run something like part 1 for NxM boards. But the idea is
# that I want to track if I'm in an infinite loop or not (BFS maybe?)


def generate_boards(board: np.ndarray):
    boards = []

    height, width = board.shape
    for i in range(height):
        for j in range(width):
            # Don't put an obstacle over an obstacle (duh)
            if board[i][j] == "#" or board[i][j] in move_rotations.keys():
                continue

            # Otherwise, create a new copy of the board as if we've put an
            # obstacle
            new_board = board.copy()
            new_board[i][j] = "#"

            boards.append(new_board)

    return boards


possible_boards = generate_boards(og_board)
p2 = 0

for i, board in enumerate(possible_boards):
    print(i)
    # Simulate each of the boards
    unique_edges = set()
    while True:
        direction, location = find_guard(board)
        board, moves, next_pos = move(board, direction, location)
        # We've found a non-obstacle, i.e. the end of the board
        if out_of_bounds(board, next_pos):
            break

        # Else, we should track if the guard is in an infinite loop by looking
        # at the times they bumped into obstacles
        match direction:
            case "^":
                final_pos = direction, next_pos
            case ">":
                final_pos = direction, next_pos
            case "v":
                final_pos = direction, next_pos
            case "<":
                final_pos = direction, next_pos
            case _:
                exit(1)

        if final_pos in unique_edges:
            p2 += 1
            break

        unique_edges.add(final_pos)

print(f"Part 2: {p2}")
