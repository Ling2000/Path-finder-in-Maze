import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from numpy.ma import masked_array
import sys


def maze(d, p):     # Generate a maze with a cell of fire
    m = np.random.binomial(1, 1 - p, size=(d, d))
    m[0, 0] = 1
    m[d - 1, d - 1] = 1
    rr = random.randrange(0, d - 1)
    cc = random.randrange(0, d - 1)
    m = m*1.0
    m[rr, cc] = 2
    # m[0,1] = 2
    print("origin fire place", "(", rr, ",", cc, ")")
    #print(m)
    if m[d - 1, d - 1] == 2 or m[0, 0] == 2:  # make sure start or end not on fire at first.
        print("origin fire place on the start or on the end no way u can out")
        sys.exit(1)  # end

    plt.matshow(m, cmap=plt.cm.gray)
    plt.text(x=cc, y=rr, s='Fire')
    plt.text(x=0, y=0, s='s')
    plt.text(x=d - 1, y=d - 1, s='g')
    plt.show()      # show the maze
    return m


def bfs(m, pred, s, e):
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
        # right
        if col + 1 < dd and m[row][col + 1] == 1 and [row, col + 1] not in closedSet and pred[row][col + 1] != 1:
            que.append((row, col + 1))
            closedSet.append([row, col + 1])
            new = (row, col + 1)
            path[new] = now  # for print path
        # down
        if row + 1 < dd and m[row + 1][col] == 1 and [row + 1, col] not in closedSet and pred[row + 1][col] != 1:
            que.append((row + 1, col))
            closedSet.append([row + 1, col])
            new = (row + 1, col)
            path[new] = now  # for print path
        # left
        if col - 1 > -1 and m[row][col - 1] == 1 and [row, col - 1] not in closedSet and pred[row][col - 1] != 1:
            que.append((row, col - 1))
            closedSet.append([row, col - 1])
            new = (row, col - 1)
            path[new] = now  # for print path
        # up
        if row - 1 > -1 and m[row - 1][col] == 1 and [row - 1, col] not in closedSet and pred[row - 1][col] != 1:
            que.append((row - 1, col))
            closedSet.append([row - 1, col])
            new = (row - 1, col)
            path[new] = now  # for print path

    if len(que) == 0:
        print("Juat a bad maze or Fire or/and obstacles are blocking you. No way out")
        # print(m)
        sys.exit(1)  # end
        return n

def advance_fire_one_step(mz, d, q):        # Returns a new maze for fire spread after one step
    mazeff = mz.copy()
    # print(q)
    for i in range(len(mazeff)):
        for j in range(len(mazeff)):
            if mz[i][j] == 1:
                k = 0
                if i > 0 and mz[i - 1][j] == 2:
                    k = k + 1
                if j > 0 and mz[i][j - 1] == 2:
                    k = k + 1
                if i < len(mazeff) - 1 and mz[i + 1][j] == 2:
                    k = k + 1
                if j < len(mazeff) - 1 and mz[i][j + 1] == 2:
                    k = k + 1
                if k == 0:
                    continue
                prob = 1 - math.pow((1 - q), k)
                if random.random() <= prob:
                    mazeff[i][j] = 2
    return mazeff


def predict_spread(maze):  # Return a new 2dlist with the fastest number of steps that can be burned to each grid
    fires = []
    for i in range(len(maze)):
        for j in range(len(maze)):
            if maze[i][j] == 2:
                fires.append([i, j])
    predict = [[-1] * len(maze) for z in range(len(maze))]    # get a new 2dlist for return
    accessed = []
    # predict, accessed = spread_once(maze, accessed, predict, fires[0], {}, [fires[0]])
    for k in range(len(fires)):
        predict, accessed = spread_once(maze, accessed, predict, fires[k], {}, [fires[k]])
    return predict



def spread_once(maze, accessed, predict, fire, prev, queue):
    fire = tuple(fire)
    # if len(queue) == 0:
    #     return predict
    if fire not in accessed:
        accessed.append(fire)
        if fire in prev:
            p = prev[fire]
            predict[fire[0]][fire[1]] = predict[p[0]][p[1]] + 1
        else:
            predict[fire[0]][fire[1]] = 0
    else:
        if fire in prev:
            p = prev[fire]
            if predict[fire[0]][fire[1]] < predict[p[0]][p[1]] + 1:
                return predict, accessed
            else:
                predict[fire[0]][fire[1]] = predict[p[0]][p[1]] + 1
        else:
            predict[fire[0]][fire[1]] = 0
    fire = list(fire)
    if fire[0] > 0 and maze[fire[0] - 1][fire[1]] == 1:
        if predict[fire[0] - 1][fire[1]] == -1 or predict[fire[0] - 1][fire[1]] > predict[fire[0]][fire[1]] + 1:  # up
            if [fire[0] - 1, fire[1]] not in queue:
                queue.append([fire[0] - 1, fire[1]])
                prev[tuple([fire[0] - 1, fire[1]])] = fire
    if fire[0] < len(maze) - 1 and maze[fire[0] + 1][fire[1]] == 1:
        if predict[fire[0] + 1][fire[1]] == -1 or predict[fire[0] + 1][fire[1]] > predict[fire[0]][fire[1]] + 1:  # down
            if [fire[0] + 1, fire[1]] not in queue:
                queue.append([fire[0] + 1, fire[1]])
                prev[tuple([fire[0] + 1, fire[1]])] = fire
    if fire[1] > 0 and maze[fire[0]][fire[1] - 1] == 1:
        if predict[fire[0]][fire[1] - 1] == -1 or predict[fire[0]][fire[1] - 1] > predict[fire[0]][fire[1]] + 1:  # left
            if [fire[0], fire[1] - 1] not in queue:
                queue.append([fire[0], fire[1] - 1])
                prev[tuple([fire[0], fire[1] - 1])] = fire
    if fire[1] < len(maze) - 1 and maze[fire[0]][fire[1] + 1] == 1:
        if predict[fire[0]][fire[1] + 1] == -1 or predict[fire[0]][fire[1] + 1] > predict[fire[0]][fire[1]] + 1:  # righ
            if [fire[0], fire[1] + 1] not in queue:
                queue.append([fire[0], fire[1] + 1])
                prev[tuple([fire[0], fire[1] + 1])] = fire
    if len(queue) != 0:
        q = queue.pop(0)
        predict, accessed = spread_once(maze, accessed, predict, q, prev, queue)
    return predict, accessed


if __name__ == "__main__":
    print("Please input in that form: dim p q")
    a = list(map(float, input(' ').split()))
    mat = maze(int(a[0]), a[1])
    sys.setrecursionlimit(1500)
    # print(mat)
    start = (0, 0)
    end = (a[0] - 1, a[0] - 1)
    mazef = mat.copy()
    predict1 = predict_spread(mazef)
    if a[2] == 0:  # When the fire does not spread, it is like an extra wall
        mazereal = bfs(mat, predict1, start, end)
        mazef = masked_array(mazef, mazef < 1.5)
        mazereal = masked_array(mazereal, mazereal > 1.5)
        plt.imshow(mazereal, interpolation='nearest', cmap=plt.cm.gray)
        plt.imshow(mazef, interpolation='nearest', cmap=plt.cm.Reds_r)
        plt.show()
        print("you reach the goal")
        sys.exit(0)  # end
    predict2 = np.random.binomial(1, 1 - 0.5, size=(int(a[0]), int(a[0])))
    predict2 = predict2*1.0
    # print(predict2)
    ii = 0
    while ii < int(a[0]):
        jj = 0
        while jj < int(a[0]):
            predict2[ii][jj] = predict1[ii][jj]
            jj = jj + 1
            pass
        ii = ii + 1                   # end for get a new matrix

    #   if a[2] <= 0.5:
    #       predict2 = predict2+0.1
    mazereal = bfs(mat, predict2, start, end)       # find the shortest path
    # print(mazereal)
    walk = [start]
    mazereal[0, 0] = 0.3
    #  accessed = [[0, 0]]
    life = True
    while life:     # walk once for one step until burned or reach the goal
        now = walk[-1]  # current point
        row, col = now
        if now == end:      # reach the goal
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
            walk.append((row, col + 1))                 # walked path
            mazef = advance_fire_one_step(mazef, int(a[0]), a[2])   # new fire situation
            predict2 = predict_spread(mazef)                        # new predict
            mazereal = bfs(mazef, predict2, (row, col + 1), end)     # new path
            mazereal[row, col + 1] = 0.3
            continue

        elif row + 1 < a[0] and mazereal[row + 1, col] == 0.5:  # down
            walk.append((row + 1, col))
            mazef = advance_fire_one_step(mazef, int(a[0]), a[2])
            predict2 = predict_spread(mazef)
            mazereal = bfs(mazef, predict2, (row + 1, col), end)
            mazereal[row + 1, col] = 0.3
            continue

        elif col - 1 > -1 and mazereal[row, col - 1] == 0.5:  # left
            walk.append((row, col - 1))
            mazef = advance_fire_one_step(mazef, int(a[0]), a[2])
            predict2 = predict_spread(mazef)
            mazereal = bfs(mazef, predict2, (row, col - 1), end)
            mazereal[row, col - 1] = 0.3
            continue

        elif row - 1 > -1 and mazereal[row - 1, col] == 0.5:  # up
            walk.append((row - 1, col))
            mazef = advance_fire_one_step(mat, int(a[0]), a[2])
            predict2 = predict_spread(mazef)
            mazereal = bfs(mazef, predict2, (row - 1, col), end)
            mazereal[row - 1, col] = 0.3
            continue
    # print(mazereal)
    # maskarray for showing the path and fire
    # print(mazef)
    mazef = masked_array(mazef, mazef < 1.5)
    mazereal = masked_array(mazereal, mazereal > 1.5)
    plt.imshow(mazereal, interpolation='nearest', cmap=plt.cm.gray)
    plt.imshow(mazef, interpolation='nearest', cmap=plt.cm.Reds_r)
    plt.show()
    # print(mazereal)
    # print(mazef)
