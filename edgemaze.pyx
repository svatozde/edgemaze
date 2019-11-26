# cython: profile=False

cimport numpy as np
import numpy as np
from numpy cimport ndarray
from numpy import ndarray
from typing import List
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
        cpdef Point point = Point(row, column)
        cpdef Point p
        if self.directions[row,column] == b' ':
            raise ValueError(
                "No target point is reachable from: [{},{}]"
                    .format(point.x,point.y)
                )

        while True:
            symbol = self.directions[point.x][point.y]
            if symbol == b' ':
                return []

            ret.append((point.x,point.y))
            p = SYMBOLS_TO_VECTORS[symbol]
            point = Point(point.x - p.x, point.y - p.y)

            if symbol == b'X':
                break

        return ret


ctypedef public struct Point:
    int x
    int y

ctypedef public struct DistancePoint:
    int x
    int y
    int distance
    Point prev


cpdef public void set_distance(DistancePoint p, int[:,:] distances):
    if distances[p.x][p.y] == -1:
        distances[p.x][p.y] = p.distance
        #print(distances[self.x][self.y])

cpdef public void set_direction(DistancePoint dp, char[:,:] directions):
    cpdef int x = dp.x - dp.prev.x
    cpdef int y = dp.y - dp.prev.y
    if  directions[dp.x][dp.y] == b' ':
        if x == 1:
             directions[dp.x][dp.y] = b'^'
        elif x == -1:
             directions[dp.x][dp.y] = b'v'
        elif y == -1:
             directions[dp.x][dp.y] = b'>'
        elif y == 1:
             directions[dp.x][dp.y] = b'<'
        else:
            directions[dp.x][dp.y] = b'X'

cpdef Point UP = Point(1, 0)
cpdef Point DOWN = Point(-1, 0)
cpdef Point LEFT = Point(0, -1)
cpdef Point RIGHT = Point(0, 1)
cpdef Point SELF = Point(0, 0)

cpdef list DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

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

#cdef Point[:] neighbours = array.array('o',[UP, DOWN, LEFT, RIGHT])

@cython.profile(False)
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




@cython.profile(False)
def flooding(start: List[SPoint], in_maze: ndarray) -> Analyzed:
    cdef int[:,:] maze = in_maze.astype(int)
    cdef list stack = [
        DistancePoint(
            s.x,
            s.y,
            0,
            Point(s.x,s.y)
            ) for s in start
        ]

    arr =  ndarray((maze.shape[0],maze.shape[1]),dtype='int')
    arr.fill(-1)
    cdef int[:,:] distances = np.ascontiguousarray(arr,np.int32)

    carr =  np.chararray((maze.shape[0],maze.shape[1]))
    carr.fill(b' ')
    cdef char[:,:] directions = np.ascontiguousarray(carr)


    for p in stack:
        set_distance(p, distances)
        set_direction(p, directions)

    cpdef DistancePoint new_point = DistancePoint(0,0,0,Point(0,0))
    cpdef DistancePoint current = DistancePoint(0,0,0,Point(0,0))
    cpdef Point direction
    cpdef Point prev

    cpdef int max_x = maze.shape[0]
    cpdef int max_y = maze.shape[1]

    while (stack):
        current = stack.pop(0)
        for d in DIRECTIONS:
            direction = d
            if can_go(current,maze,distances,direction,max_x,max_y):
                new_point=DistancePoint(
                    current.x + direction.x,
                    current.y + direction.y,
                    current.distance + 1,
                    Point(current.x, current.y)
                )

                set_distance(new_point, distances)
                set_direction(new_point, directions)
                stack.append(new_point)

    return Analyzed(
            distances,
            directions,
            is_reachable(distances)
        )


def is_reachable(dist):
    unique, counts = np.unique(dist, return_counts=True)
    d = dict(zip(unique, counts))
    return -1 not in d


cdef int is_in_bounds(Point p , int[:,:] a):
    return 0 <= p.x < a.shape[0] and 0 <= p.y < a.shape[1]

@cython.profile(False)
cdef int can_go(
        DistancePoint p,
        int[:,:]  maze,
        int[:,:]  distances,
        Point direction,
        int max_x,
        int max_y
        ):

    #print("can go 1")
    cdef Point to = Point(p.x + direction.x, p.y + direction.y)

    if not (0 <= to.x < max_x and 0 <= to.y < max_y):
        return False

    cdef int val = 0
    if (direction.x == UP.x ) or (direction.y == RIGHT.y):
        val = maze[to.x][to.y]
    else:
        val = maze[p.x,p.y]

    if direction.x != 0:
        bit = GO_UP_BIT
    else:
        bit = GO_LEFT_BIT

    cdef int wall = is_nth_bit_set(val, bit)
    cdef int distance = distances[to.x][to.y]
    return distance < 0 and not wall > 0


def get_from_np_array(p: Point, maze: ndarray):
    return maze[p.x,p.y]

def analyze(maze: ndarray) -> PyAnalyzed:
    if maze is None or maze.ndim != 2:
        raise TypeError("Input mazr must not be null and its dimension"
                        "must be exactly 2.")
    start = find_start(maze)
    a = flooding(start, maze)

    return PyAnalyzed(
        np.asarray(a.distances),
        np.asarray(a.directions, dtype=np.object),
        a.is_reachable
        )
