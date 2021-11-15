import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from numpy.ma import masked_array
import sys


def maze(d, p):  # Generate a maze with a cell of fire
    m = np.random.binomial(1, 1 - p, size=(d, d))
    m[0, 0] = 1
    m[d - 1, d - 1] = 1
    rr = random.randrange(0, d - 1)
    cc = random.randrange(0, d - 1)
    m[rr, cc] = 2
    print("origin fire place", "(", rr, ",", cc, ")")

    if m[d - 1, d - 1] == 2 or m[0, 0] == 2:  # make sure start or end not on fire at first.
        print("origin fire place on the start or on the end no way u can out")
        sys.exit(1)  # end

    plt.matshow(m, cmap=plt.cm.gray)
    plt.text(x=cc, y=rr, s='Fire')
    plt.text(x=0, y=0, s='s')
    plt.text(x=d - 1, y=d - 1, s='g')
    plt.show()
    return m

def bfs(m, s, e):
    rr, cc = e  # get the row and col for end point
    dd = rr + 1  # get the dimension
    que = [s]
    sl = list(s)
    closedSet = [sl]  # accessed points
    path = {s: s}  # use dict  to record the short path
    n = np.matlib.rand(int(a[0]), int(a[0]))
    i = 0
    while i < int(a[0]):
        j = 0
        while j < int(a[0]):
            n[i, j] = m[i, j]
            j = j + 1
            pass
        i = i + 1

    while que:  # lst not null (if null that means no way out)
        now = que.pop(0)  # pop the front and new=the popped one
        if now == e:
            # print("shortest path out")
            tr = rr
            tc = cc
            sr,sc = s
            while tr != sr or tc != sc:  # use dict  to record the short path from end to start until reach (0,0)
                way = path[(tr,tc)]
                rrr, ccc = way
                n[rrr, ccc] = 0.5  # the way out
                tr,tc=path[(tr,tc)]  # value become the key now
                # print(path[(tr,tc)])
            n[int(rr), int(cc)] = 0.5
            # plt.matshow(n, cmap=plt.cm.gray)
            # plt.show()
            return n

        row, col = now

        if col + 1 < dd and m[row][col + 1] == 1 and [row, col + 1] not in closedSet:  # right
            que.append((row, col + 1))
            closedSet.append([row, col + 1])
            new = (row, col + 1)
            path[new] = now  # for print path

        if row + 1 < dd and m[row + 1][col] == 1 and [row + 1, col] not in closedSet:  # down
            que.append((row + 1, col))
            closedSet.append([row + 1, col])
            new = (row + 1, col)
            path[new] = now  # for print path

        if col - 1 > -1 and m[row][col - 1] == 1 and [row, col - 1] not in closedSet:  # left
            que.append((row, col - 1))
            closedSet.append([row, col - 1])
            new = (row, col - 1)
            path[new] = now  # for print path

        if row - 1 > -1 and m[row - 1][col] == 1 and [row - 1, col] not in closedSet:  # up
            que.append((row - 1, col))
            closedSet.append([row - 1, col])
            new = (row - 1, col)
            path[new] = now  # for print path

    if len(que) == 0:
        print("Juat a bad maze or Fire or/and obstacles are blocking you. No way out")
        sys.exit(1)  # end
        return n


def advance_fire_one_step(mz, d, q):    # Returns a new maze for fire spread after one step
    mazef = mz.copy()
    #   print(q)
    for i in range(len(mazef)):
        for j in range(len(mazef)):
            if mz[i][j] == 1:
                k = 0
                if i > 0 and mz[i - 1][j] == 2:
                    k = k + 1
                if j > 0 and mz[i][j - 1] == 2:
                    k = k + 1
                if i < len(mazef) - 1 and mz[i + 1][j] == 2:
                    k = k + 1
                if j < len(mazef) - 1 and mz[i][j + 1] == 2:
                    k = k + 1
                if k == 0:
                    continue
                prob = 1 - math.pow((1 - q), k)
                if random.random() <= prob:
                    mazef[i][j] = 2
    return mazef


if __name__ == "__main__":
    print("Please input in that form: dim p q")
    a = list(map(float, input(' ').split()))
    mat = maze(int(a[0]), a[1])
    # print(mat)
    start = (0, 0)
    end = (a[0] - 1, a[0] - 1)
    mazef = mat.copy()      # get a new maze in case u want to print the origin one
    mazereal = bfs(mat, start, end)      # find the shortest path
    walk = [start]
    mazereal[0,0] = 0.3
    #  accessed = [[0, 0]]
    life = True
    while life:      # walk once for one step until burned or reach the goal
        now = walk[-1]  # current point
        row, col = now
        if now == end:       # reach the goal
            print("you reach the goal")
            while walk:
                way = walk.pop(0)
                rrr, ccc = way
                mazereal[rrr, ccc] = 0.3  # the way out

            # plt.matshow(mazereal, cmap=plt.cm.gray)
            # plt.show()
            break

        if mazef[row][col] == 2:        # burned
            life = False
            while walk:
                now = walk.pop()
                rrr, ccc = now
                mazereal[rrr, ccc] = 0.3
                # plt.matshow(mazereal, cmap=plt.cm.gray)
                # plt.show()
            print("u have been burned")
            print("in", row, col)
            break

        if col + 1 < a[0] and mazereal[row, col + 1] == 0.5:  # right
            walk.append((row, col + 1))
            mazef = advance_fire_one_step(mazef, int(a[0]), a[2])
            mazereal = bfs(mazef, (row, col + 1), end)
            mazereal[row, col + 1] = 0.3
            continue

        elif row + 1 < a[0] and mazereal[row + 1, col] == 0.5:  # down
            walk.append((row + 1, col))
            mazef = advance_fire_one_step(mazef, int(a[0]), a[2])
            mazereal = bfs(mazef, (row + 1, col), end)
            mazereal[row + 1, col] = 0.3
            continue

        elif col - 1 > -1 and mazereal[row, col - 1] == 0.5:  # left
            walk.append((row, col - 1))
            mazef = advance_fire_one_step(mazef, int(a[0]), a[2])
            mazereal = bfs(mazef, (row, col - 1), end)
            mazereal[row, col - 1] = 0.3
            continue

        elif row - 1 > -1 and mazereal[row - 1, col] == 0.5:  # up
            walk.append((row - 1, col))
            mazef = advance_fire_one_step(mat, int(a[0]), a[2])
            mazereal = bfs(mazef, (row - 1, col), end)
            mazereal[row - 1, col] = 0.3
            continue
    # maskarray for showing the path and fire
    mazef = masked_array(mazef, mazef < 1.5)
    mazereal = masked_array(mazereal, mazereal > 1.5)
    plt.imshow(mazereal, interpolation='nearest', cmap=plt.cm.gray)
    plt.imshow(mazef, interpolation='nearest', cmap=plt.cm.Reds_r)
    plt.show()

    # print(mazef)