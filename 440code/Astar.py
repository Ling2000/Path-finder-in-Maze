import threading
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import math
import imageio
def create_gif(image_list, gif_name, duration):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    n=0
    f=imageio.imread(image_list[-1])
    while n<20:
        frames.append(f)
        n=n+1
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def maze(d,p):
    m = np.random.binomial(1,1-p,size=(d,d))
    m[0,0]=1
    m[d-1,d-1]=1
    plt.matshow(m, cmap=plt.cm.gray)
    plt.text(x=0, y=0, s='s')
    plt.text(x=d-1, y=d-1, s='g')
    plt.show()
    return m

def takeSecond(elem):
    return elem[1]

def A_star(m,d):
    #this part is used to generate a float maze
    n=np.matlib.rand(d,d)
    i=0
    while i<int(a[0]):
        j=0
        while j<int(a[0]):
            n[i,j]=mat[i,j]
            j=j+1
            pass
        i=i+1
    fringe = [([0,0],math.sqrt(2*d*d))]
    closedSet = [[0,0]]

    prev={(0,0):(0,0)}
    p=["maze0.png"]
    #cou=0
    while len(fringe) > 0:
        currentState = fringe.pop(-1)
        cs, pri = currentState
        row, col= cs

        #0.5 means this node has been explored
        n[row,col]=0.5

        #this part is used to generate gif. If you want to generate, pleased get rid of or # before code
        #plt.matshow(n, cmap=plt.cm.gray)
        #filename = "./maze" + str(cou) + ".png"
        #plt.savefig(filename)
        #fn=filename.lstrip('./')
        #p.append(fn)
        #cou=cou+1
        #if we reach the goal
        if cs == [d-1,d-1]:
            #use prev to generate the road
            pre=prev[(d-1,d-1)]
            while pre!=(0,0):
                l,h=pre
                n[l,h]=0.3
                pre=prev[pre]
                pass
            n[0,0]=0.3
            n[d-1,d-1]=0.3
            plt.matshow(n, cmap=plt.cm.gray)

            #this part is used to generate gif. If you want to generate, pleased get rid of or # before code
            #filename = "./maze" + str(cou+1) + ".png"
            #plt.savefig(filename)
            #fn=filename.lstrip('./')
            #p.append(fn)
            #create_gif(p,'./maze_a*.gif',  0.35)

            plt.show()
            return True
        #this part is used to compute the piority
        U=math.sqrt(abs((row-d)*(row-d))+abs((col-d+1)*(col-d+1)))+math.sqrt(abs((row - 1)*(row - 1))+abs(col*col))
        L=math.sqrt(abs((row-d+1)*(row-d+1))+abs((col-d)*(col-d)))+math.sqrt(abs(row*row)+abs((col-1)*(col-1)))
        R=math.sqrt(abs((row-d+1)*(row-d+1))+abs((col-d+2)*(col-d+2)))+math.sqrt(abs((row + 1)*(row + 1))+abs(col*col))
        D=math.sqrt(abs((row-d+2)*(row-d+2))+abs((col-d+1)*(col-d+1)))+math.sqrt(abs(row*row)+abs((col+1)*(col+1)))

        if row-1>-1 and m[row - 1][col] == 1 and [row - 1, col] not in closedSet:  # up
            n[row-1,col]=0.8
            fringe.append(([row - 1, col],U))
            prev[(row - 1, col)]=(row,col)
            closedSet.append([row-1,col])

        if col-1>-1 and m[row][col - 1] == 1 and [row, col - 1] not in closedSet:  # left
            n[row,col-1]=0.8
            fringe.append(([row, col - 1],L))
            prev[(row, col - 1)]=(row,col)
            closedSet.append([row,col-1])

        if col+1<d and m[row][col + 1] == 1 and [row, col + 1] not in closedSet:  # right
            n[row,col+1]=0.8
            fringe.append(([row, col + 1],R))
            prev[(row, col + 1)]=(row,col)
            closedSet.append([row,col+1])

        if row+1<d and m[row + 1][col] == 1 and [row + 1, col] not in closedSet:  # down
            n[row+1,col]=0.8
            fringe.append(([row + 1, col],D))
            prev[(row + 1, col)]=(row,col)
            closedSet.append([row+1,col])

        closedSet.append(cs)
        #sort base on the piority
        fringe.sort(key=takeSecond,reverse=True)
        if len(fringe) == 0:
            break
    return False

if __name__ == "__main__":
    print("Please input in that form: dim p")
    a = list(map(float, input(' ').split()))
    #generate maze
    mat=maze(int(a[0]),a[1])
    time_begin = time.time()
    #a*
    if A_star(mat,int(a[0])):
        print("A*: This maze is reachable")
    else:
        print("A*: This maze is NOT reachable")
    time_end = time.time()
    timme = time_end - time_begin
    print('time:', timme)
    print("Done.")
