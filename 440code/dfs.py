# DFS
import threading
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

def maze(d, p):     # Generate a maze with a cell of fire
    m = np.random.binomial(1, 1 - p, size=(d, d))
    m[0, 0] = 1
    m[d - 1, d - 1] = 1
    plt.matshow(m, cmap=plt.cm.gray)
    plt.text(x=0, y=0, s='s')
    plt.text(x=d - 1, y=d - 1, s='g')
    plt.show()
    return m


def dfs(m, s, e):
    rr,cc = e     #get the row and col for end point
    dd = rr+1     #get the dimension
    stack = [s]
    closedSet = [[0,0]] #accessed points
    n = np.matlib.rand(int(a[0]), int(a[0]))        # get a new float type matrix for showing path
    i = 0
    while i < int(a[0]):
        j = 0
        while j < int(a[0]):
            n[i, j] = mat[i, j]
            j = j + 1
            pass
        i = i + 1
    while stack:  # lst not null (if null that means no way out)
        now = stack[-1]  # current point
        if now == e:
            print("you reach the goal")
            while stack:
                way = stack.pop(0)
                rrr, ccc = way
                n[rrr, ccc] = 0.5  # the way out
            # print(stack)
            plt.matshow(n, cmap=plt.cm.gray)
            plt.show()
            # print(n)
            return True

        row, col = now
        if col+1 < dd and m[row][col + 1] == 1 and [row, col+1] not in closedSet:  # right
            stack.append((row, col + 1))
            closedSet.append([row, col+1])
            continue

        elif row+1 < dd and m[row + 1][col] == 1 and [row+1, col] not in closedSet:  # down
            stack.append((row + 1, col))
            closedSet.append([row+1, col])
            continue

        elif col-1 > -1 and m[row][col - 1] == 1 and [row, col-1] not in closedSet:  # left
            stack.append((row, col - 1))
            closedSet.append([row, col-1])
            continue

        elif row-1 > -1 and m[row - 1][col] == 1 and [row-1, col] not in closedSet:  # up
            stack.append((row - 1, col))
            closedSet.append([row-1, col])
            continue

        else:  # back
            stack.pop()  # pop the last point

    if len(stack) == 0:
        print("no way out")
        return False


if __name__ == "__main__":
    print("Please input in that form: dim p")
    a = list(map(float, input(' ').split()))
    mat = maze(int(a[0]), a[1])
    # print(mat)
    start = (0, 0)
    end = (a[0] - 1, a[0] - 1)
    time_begin = time.time()
    if dfs(mat, start, end):
        print("DFS: This maze is reachable")
    else:
        print("DFS: This maze is NOT reachable")
    time_end = time.time()
    timme = time_end - time_begin
    print('time:', timme)
