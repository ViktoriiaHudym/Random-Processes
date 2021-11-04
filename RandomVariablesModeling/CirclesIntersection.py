import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations


file = 'circles.txt'

def read_file(filename):
    with open(filename) as f:
        array = []
        for line in f:
            array.append([float(x) for x in line.split()])
    return array


circles_data = read_file(file)

n = len(circles_data)
k = 2
comb = math.comb(n, k)
comb_list = list(combinations(circles_data, k))

intersect_bool = []
for circles_pair in comb_list:
    x1, y1, r1 = circles_pair[0]
    x2, y2, r2 = circles_pair[1]

    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    if distance <= r1 + r2:
        intersect_bool.append(True)
    else:
        intersect_bool.append(False)


# 1 - circles and their intersection + square around the smallest circle
# 2 - '1' + randomly generated points inside the square

def graph(data, graph_type=None, smallest=None, points=None):
    figure, ax = plt.subplots()
    ax.set_aspect('equal')
    for circle in data:
        ax.add_artist(plt.Circle((circle[0], circle[1]), circle[2],
                                 color='r', alpha=.2))
    ax.set_xlim(-3, 8)
    ax.set_ylim(-3, 8)
    ax.grid(linewidth=0.2)

    if graph_type == '1':
        side = 2 * smallest[2]
        x_lb = smallest[0] - side / 2
        y_lb = smallest[1] - side / 2
        square = plt.Rectangle((x_lb, y_lb), side, side,
                               fc='none', ec='black', lw=0.5)
        ax.add_artist(square)

    if graph_type == '2':
        x, y = list(zip(*points))
        plt.scatter(x, y, color='black', s=0.05)

    plt.show()


def MonteCarlo():
    graph(circles_data)

    def find_smallest_circle(data):
        R = []
        for i in range(len(data)):
            R.append(data[i][2])
        min_radius = min(R)

        return data[R.index(min_radius)]

    def generate_points(a, b, radius, n):
        xr = np.random.uniform(a - radius, a + radius, n)
        yr = np.random.uniform(b - radius, b + radius, n)
        return list(zip(xr, yr))

    def points_in_intersection(xy, data):
        result = []
        x_point, y_point = list(zip(*xy))
        center_x, center_y, radius = list(zip(*data))
        for i in range(len(x_point)):
            result.append([x_point[i], y_point[i]])
            for j in range(len(data)):
                if ((x_point[i] - center_x[j])**2 + (y_point[i] - center_y[j])**2) >= (radius[j] ** 2):
                    result.pop(-1)
                    break
        return result

    smallest_circle = find_smallest_circle(circles_data)
    x, y, r = smallest_circle
    graph(circles_data, '1', smallest=smallest_circle)

    m = 1000000
    xy_random = generate_points(x, y, r, m)
    graph(circles_data, '2', points=xy_random)

    inside_intersection = points_in_intersection(xy_random, circles_data)
    points_amount = len(inside_intersection)
    S = (2 * r) ** 2
    s = S * points_amount / m

    print(f'\nThe radius of the smallest circle is equal to: {r}\n'
          f'The area of the square is equal to: {S}\n'
          f'Out of {m} points, {points_amount} got inside the intersection of the circles\n'
          f'The area of the circles intersection is equal to: {s}')


if __name__ == '__main__':
    if all(intersect_bool):
        print(f'\nThe circles intersect')
        MonteCarlo()
    else:
        print(f'\nThe circles NOT intersect')
        exit(1)
