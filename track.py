import os
import numpy as np

t = "track1.txt"
f = os.path.join(os.getcwd(), t)

def convert_trackfile(trackfile):
    lines = []

    with open(f) as file:
        lines = file.readlines()

    return lines

def print_track(lines):
    for l in lines:
        print(l, end="")

def build_track(lines):
    rows = len(lines)
    cols = len(lines[0])
    track_to_numpy = []

    for row in range(len(lines)):
        r = []
        for col in range(len(lines[row])):
            if lines[row][col] != "\n":
                r.append(lines[row][col])
        track_to_numpy.append(r)

    return np.asarray(track_to_numpy)
