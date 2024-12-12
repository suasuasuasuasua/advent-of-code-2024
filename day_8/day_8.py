from __future__ import annotations

"""
--- Day 8: Resonant Collinearity ---

You find yourselves on the roof of a top-secret Easter Bunny installation.

While The Historians do their thing, you take a look at the familiar huge
antenna. Much to your surprise, it seems to have been reconfigured to emit a
signal that makes people 0.1% more likely to buy Easter Bunny brand Imitation
Mediocre Chocolate as a Christmas gift! Unthinkable!

Scanning across the city, you find that there are actually many such antennas.
Each antenna is tuned to a specific frequency indicated by a single lowercase
letter, uppercase letter, or digit. You create a map (your puzzle input) ofh
these antennas. For example:

............
........0...
.....0......
.......0....
....0.......
......A.....
............
............
........A...
.........A..
............
............

The signal only applies its nefarious effect at specific antinodes based on the
resonant frequencies of the antennas. In particular, an antinode occurs at any
point that is perfectly in line with two antennas of the same frequency - but
only when one of the antennas is twice as far away as the other. This means that
for any pair of antennas with the same frequency, there are two antinodes, one
on either side of them.

So, for these two antennas with frequency a, they create the two antinodes
marked with #:

..........
...#......
..........
....a.....
..........
.....a....
..........
......#...
..........
..........

Adding a third antenna with the same frequency creates several more antinodes.
It would ideally add four antinodes, but two are off the right side of the map,
so instead it adds only two:

..........
...#......
#.........
....a.....
........a.
.....a....
..#.......
......#...
..........
..........

Antennas with different frequencies don't create antinodes; A and a count as
different frequencies. However, antinodes can occur at locations that contain
antennas. In this diagram, the lone antenna with frequency capital A creates no
antinodes but has a lowercase-a-frequency antinode at its location:

..........
...#......
#.........
....a.....
........a.
.....a....
..#.......
......A...
..........
..........

The first example has antennas with two different frequencies, so the antinodes
they create look like this, plus an antinode overlapping the topmost A-frequency
antenna:

......#....#
...#....0...
....#0....#.
..#....0....
....0....#..
.#....A.....
...#........
#......#....
........A...
.........A..
..........#.
..........#.

Because the topmost A-frequency antenna overlaps with a 0-frequency antinode,
there are 14 total unique locations that contain an antinode within the bounds
of the map.

--- Part One ---

Calculate the impact of the signal. How many unique locations within the bounds
of the map contain an antinode?

--- Part Two ---

Watching over your shoulder as you work, one of The Historians asks if you took
the effects of resonant harmonics into your calculations.

Whoops!

After updating your model, it turns out that an antinode occurs at any grid
position exactly in line with at least two antennas of the same frequency,
regardless of distance. This means that some of the new antinodes will occur at
the position of each antenna (unless that antenna is the only one of its
frequency).

So, these three T-frequency antennas now create many antinodes:

T....#....
...T......
.T....#...
.........#
..#.......
..........
...#......
..........
....#.....
..........

In fact, the three T-frequency antennas are all exactly in line with two
antennas, so they are all also antinodes! This brings the total number of
antinodes in the above example to 9.

The original example now has 34 antinodes, including the antinodes that appear
on every antenna:

##....#....#
.#.#....0...
..#.#0....#.
..##...0....
....0....#..
.#...#A....#
...#..#.....
#....#.#....
..#.....A...
....#....A..
.#........#.
...#......##

Calculate the impact of the signal using this updated model. How many unique
locations within the bounds of the map contain an antinode?
"""


import re
from collections import defaultdict
from pathlib import Path

import numpy as np

input_file = Path("sample.txt")
input_file = Path("input.txt")

with open(input_file, "r") as file_in:
    data = [[c for c in row.strip()] for row in file_in.readlines()]

data = np.array(data)


class Antenna:
    def __init__(self, i: int, j: int):
        self.i = i
        self.j = j

    def diff(self, other: Antenna) -> Antenna:
        new_j = self.j - other.j
        new_i = self.i - other.i

        return Antenna(new_i, new_j)

    def __add__(self, other: Antenna):
        new_j = self.j + other.j
        new_i = self.i + other.i

        return Antenna(new_i, new_j)

    def __mul__(self, a: int):
        new_j = self.j * a
        new_i = self.i * a

        return Antenna(new_i, new_j)

    def __lt__(self, other: Antenna) -> bool:
        return (self.i, self.j) < (other.i, other.j)

    def __eq__(self, other: Antenna) -> bool:
        if not isinstance(other, Antenna):
            return False

        return self.i == other.i and self.j == other.j

    def __hash__(self):
        return hash((self.i, self.j))

    def __str__(self) -> str:
        return f"{self.j=};{self.i=}"

    def __repr__(self) -> str:
        return str(self)


antennas: dict[list[Antenna]] = defaultdict(list)
antenna_pattern = re.compile(r"([a-zA-Z\d])")

for j, row in enumerate(data):
    for i, col in enumerate(row):
        if match := antenna_pattern.match(col):
            # Grab the letter
            antenna = match.group(1)
            # Save the index position
            pos = Antenna(i, j)

            antennas[antenna].append(pos)


def is_in_bounds(data: np.ndarray, pos: Antenna) -> bool:
    height, width = data.shape
    j, i = pos.j, pos.i

    return 0 <= j < height and 0 <= i < width


p1_locs = set()
p2_locs = set()

for antenna, positions in antennas.items():
    # Compare the current position with all other positions
    # See if any of the antinodes are possible
    for i, pos in enumerate(positions):
        prev, rest = positions[:i], positions[i + 1 :]
        others = prev + rest

        for other in others:
            difference = pos.diff(other)
            antinode = pos + difference

            if is_in_bounds(data, antinode):
                # Add the tuple cause sets don't play nice with my class :(
                p1_locs.add(antinode)

            # Count the current antenna
            multiplier = 0
            # Iteratively extend the antinodes by multiplying outwards
            while True:
                antinode = (difference * multiplier) + pos
                # Stop when we're out of bounds
                if not is_in_bounds(data, antinode):
                    break

                p2_locs.add(antinode)
                multiplier += 1


p1 = len(p1_locs)
print(f"Part 1: {p1}")

p2 = len(p2_locs)
print(f"Part 2: {p2}")
