# cython: profile=True

cimport numpy as np
import numpy as np
from numpy cimport ndarray
from numpy import ndarray
from typing import List, Tuple
import cython

GO_LEFT_BIT = 1
GO_UP_BIT = 2


cdef class SPoint:
    cdef public int x
    cdef public int y

    def __cinit__(self, int x, int y):
        self.x = x
        self.y = y

cdef class PyAnalyzed:
    cdef public ndarray distances
    cdef public ndarray directions
    cdef public bint is_reachable

    def __cinit__(self, ndarray distances, ndarray directions,
               bint is_reachable):
        self.distances = distances
        self.directions = directions
        self.is_reachable = is_reachable



    cpdef public list path(self, int row ,  int column):
        ret = []

        cdef tuple p
        if self.directions[row,column] == b' ':
            raise ValueError(
                "No target point is reachable from: [{},{}]"
                    .format(row,column)
                )

        cdef int px = row
        cdef int py = column
        while True:
            symbol = self.directions[px,py]
            if symbol == b' ':
                return []

            ret.append((px,py))
            p = SYMBOLS_TO_VECTORS[symbol]
            px = px - p[0]
            py = py - p[1]

            if symbol == b'X':
                break

        return ret

cdef tuple UP = (1, 0)
cdef tuple DOWN = (-1, 0)
cdef tuple LEFT = (0, -1)
cdef tuple RIGHT = (0, 1)
cdef tuple SELF = (0, 0)

cdef list DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

DOWNS_S = b'v'
UP_S = b'^'
LEFT_S = b'>'
RIGHT_S = b'<'
START_S = b'X'

SYMBOLS_TO_VECTORS = {
    DOWNS_S: DOWN,
    UP_S: UP,
    LEFT_S: LEFT,
    RIGHT_S: RIGHT,
    START_S: SELF
    }

cdef int is_nth_bit_set( int x, int n):
    return x & (1 << n)

def py_is_nth_bit_set(x: int, n: int) -> int:
    return x & (1 << n)

def find_start(m: ndarray) -> List[SPoint]:
    return [
        SPoint(t[0], t[1]) for t in np.dstack(np.where(py_is_nth_bit_set(m,0)))[0]
        ]



cdef class Analyzed:
    cdef public int[:,:] distances
    cdef public char[:,:] directions
    cdef public bint is_reachable

    def __cinit__(self, int[:,:] distances, char[:,:] directions,
                   bint is_reachable):
        self.distances = distances
        self.directions = directions
        self.is_reachable = is_reachable

@cython.wraparound(False)
@cython.cdivision(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cdef Analyzed flooding(list start, ndarray in_maze):
    cdef int[:,:] maze = in_maze.astype(int)
    cdef list stack = [
            (
                s.x,
                s.y,
                0,
                s.x,
                s.y
            )
            for s in start
        ]

    arr =  ndarray((maze.shape[0],maze.shape[1]),dtype='int')
    arr.fill(-1)
    cdef int[:,:] distances = np.ascontiguousarray(arr,np.int32)

    carr =  np.chararray((maze.shape[0],maze.shape[1]))
    carr.fill(b' ')
    cdef char[:,:] directions = np.ascontiguousarray(carr)


    for p in stack:
        distances[p[0],p[1]] = 0
        directions[p[0],p[1]] = b'X'

    cdef tuple n_pt = (0,0,0,0,0)
    cdef tuple current = (0,0,0,0,0)

    cdef int max_x = maze.shape[0]
    cdef int max_y = maze.shape[1]

    cdef int x
    cdef int y

    cdef tuple d

    while (stack):
        current = stack.pop(0)
        for i in range(4):
            d = DIRECTIONS[i]
            if can_go(
                    current[0],
                    current[1],
                    maze,
                    distances,
                    d[0],
                    d[1],
                    max_x,
                    max_y
                    ):

                n_pt=(
                    current[0] + d[0],
                    current[1] + d[1],
                    current[2] + 1,
                    current[0],
                    current[1]
                )

                distances[n_pt[0]][n_pt[1]] = n_pt[2]

                x = n_pt[0] - n_pt[3]
                y = n_pt[1] - n_pt[4]
                if x == 1:
                     directions[n_pt[0]][n_pt[1]] = b'^'
                elif x == -1:
                     directions[n_pt[0]][n_pt[1]] = b'v'
                elif y == -1:
                     directions[n_pt[0]][n_pt[1]] = b'>'
                elif y == 1:
                     directions[n_pt[0]][n_pt[1]] = b'<'
                else:
                    directions[n_pt[0],n_pt[1]] = b'X'

                stack.append(n_pt)

    return Analyzed(
            distances,
            directions,
            is_reachable(distances)
        )


def is_reachable(dist):
    unique, counts = np.unique(dist, return_counts=True)
    d = dict(zip(unique, counts))
    return -1 not in d

@cython.wraparound(False)
@cython.cdivision(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cdef bint can_go(
        int x,
        int y,
        int[:,:]  maze,
        int[:,:]  distances,
        int dx,
        int dy,
        int max_x,
        int max_y
        ):

    #print("can go 1")
    cdef int tox = x + dx
    cdef int toy = y + dy

    if not (0 <= tox < max_x and 0 <= toy < max_y):
        return False

    cdef int val = 0
    if (dx == 1 ) or (dy == 1):
        val = maze[tox,toy]
    else:
        val = maze[x,y]

    if dx != 0:
        bit = GO_UP_BIT
    else:
        bit = GO_LEFT_BIT

    return distances[tox,toy] == -1 and not is_nth_bit_set(val, bit) > 0

cdef PyAnalyzed convert(Analyzed a):
    return PyAnalyzed(
        np.asarray(a.distances),
        np.asarray(a.directions, dtype=np.object),
        a.is_reachable
        )

def analyze(maze: ndarray) -> PyAnalyzed:
    if maze is None or maze.ndim != 2:
        raise TypeError("Input mazr must not be null and its dimension"
                        "must be exactly 2.")
    start = find_start(maze)
    a = flooding(start, maze)

    return convert(a)