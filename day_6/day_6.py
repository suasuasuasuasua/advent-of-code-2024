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
    board = [[c for c in row.strip()] for row in file_in.readlines()]

board = np.array(board)

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
            ys = ys[ys < y]

            # If we have any viable obstacles, set the next locations
            if ys.size:
                # The X (LR) is the same since we're in the same file still
                next_x = x
                # The Y (UD) will be one unit away from the closest obstacle
                next_y = ys[-1] + 1

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
            ys = ys[ys > y]

            # If we have any viable obstacles, set the next locations
            if ys.size:
                # The X (LR) is the same since we're in the same file still
                next_x = x
                # The Y (UD) will be one unit away from the closest obstacle
                next_y = ys[0] - 1

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
            xs = xs[xs > x]

            # If we have any viable obstacles, set the next locations
            if xs.size:
                # The X (LR) will be one unit away from the closest
                next_x = xs[0] - 1
                # The Y (UD) is the same since we're in the same row still
                next_y = y

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
            xs = xs[xs < x]

            # If we have any viable obstacles, set the next locations
            if xs.size:
                # The X (LR) will be one unit away from the closest
                next_x = xs[-1] + 1
                # The Y (UD) is the same since we're in the same row still
                next_y = y

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


def out_of_bounds(board: np.ndarray, next_y: int, next_x: int) -> bool:  #
    height, width = board.shape

    return next_y >= height or next_y < 0 or next_x >= width or next_x < 0


def move(
    board: np.ndarray, direction: str, location: tuple[int, int]
) -> tuple[np.ndarray, set[tuple[int, int]], bool]:
    # Find the span between the current position and the next obstacle ("#")
    y, x = location
    next_y, next_x, moves = find_next_pos(board, direction, y, x)

    # We've found a non-obstacle
    if out_of_bounds(board, next_y, next_x):
        return board, moves, True

    # Erase the previous position
    board[y, x] = "."
    # Move the guard and rotate 90 degrees
    board[next_y, next_x] = move_rotations[direction]

    return board, moves, False


# Part 1: Find all distinct location for the guard
distinct_moves = set()
while True:
    direction, location = find_guard(board)
    board, moves, end = move(board, direction, location)

    distinct_moves |= moves
    if end:
        break

p1 = len(distinct_moves)

print(f"Part 1: {p1}")
