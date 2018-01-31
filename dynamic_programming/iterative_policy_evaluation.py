import numpy as np
import matplotlib.pyplot as plt
from .grid_world import standard_grid

small_enough = 1e-3  # threshold for convergence


def print_values(V, g):
    for i in range(g.width):
        print("----------------------------")
        for j in range(g.height):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" %v, end="")
            else:
                print(" %.2f|" % v, end="")  # -ve take an extra space
        print()


def print_policy(P, g):
    for i in range(g.width):
        print("----------------------------")
        for j in range(g.height):
            a = P.get((i, j), ' ')
            print("  %s  |" %a)
        print()