# cython: profile=True

from dataclasses import dataclass
cimport numpy as np
import numpy as np
from typing import NamedTuple
from numpy cimport ndarray
from numpy import ndarray
from typing import Tuple, List
import cython
from numpy import int32
from numpy cimport int32_t
from cpython cimport array

GO_LEFT_BIT = 1
GO_UP_BIT = 2


cdef class Point:
    cdef public int x
    cdef public  int y

    def __cinit__(self, int x, int y):
        self.x = x
        self.y = y

    cpdef public Point add(self,Point b):
        return  Point(self.x + b.x, self.y + b.y)

    cpdef public Point minus(self,Point b):
        return  Point(self.x - b.x, self.y - b.y)

    def to_tuple(self):
        return self.x ,self.y


cdef class DistancePoint:
    cdef public int x
    cdef public int y
    cdef public int distance
    cdef public Point prev

    def __cinit__(self, int x, int y,int distance,Point prev):
        self.x = x
        self.y = y
        self.distance = distance
        self.prev = prev

    cpdef public Point add(self, Point o):
        return Point(self.x+o.x,self.y+o.y)

    cpdef public Point minus(self, Point o):
        return Point(self.x-o.x,self.y-o.y)

    cpdef public Point get_point(self):
        return Point(self.x, self.y)

    cpdef public void set_distance(self, ndarray distances):
        if distances[self.x,self.y] == -1:
            distances[self.x,self.y] = self.distance

    cpdef public void set_direction(self, ndarray directions):
        if  directions[self.x,self.y] == b' ':
            p = self.minus(self.prev)
            if p.x == 1:
                 directions[self.x,self.y] = b'^'
            elif p.x == -1:
                 directions[self.x,self.y] = b'v'
            elif p.y == -1:
                 directions[self.x,self.y] = b'>'
            elif p.y == 1:
                 directions[self.x,self.y] = b'<'
            else:
                directions[self.x,self.y] = b'X'


    cpdef DistancePoint new_distance_point(self, Point direction):
        return DistancePoint(
            x=self.x + direction.x,
            y=self.y + direction.y,
            distance=self.distance + 1,
            prev=self.get_point()
            )


UP = Point(1, 0)
DOWN = Point(-1, 0)
LEFT = Point(0, -1)
RIGHT = Point(0, 1)
SELF = Point(0, 0)


DIRECTION_SYMBOLS = {
    DOWN: b'v',
    UP: b'^',
    LEFT: b'>',
    RIGHT: b'<',
    SELF: b'X'
    }

DIRECTION_BITS = {
    DOWN: GO_UP_BIT,
    UP: GO_UP_BIT,
    LEFT: GO_LEFT_BIT,
    RIGHT: GO_LEFT_BIT
    }

SYMBOLS_TO_VECTORS = { v:k for k,v in DIRECTION_SYMBOLS.items() }

neighbours = [UP, DOWN, LEFT, RIGHT]

@cython.profile(False)
cdef int is_nth_bit_set( int x, int n):
    return x & (1 << n)

def py_is_nth_bit_set(x: int, n: int) -> int:
    return x & (1 << n)

def find_start(m: ndarray) -> List[Point]:
    return [
        Point(t[0], t[1]) for t in np.dstack(np.where(py_is_nth_bit_set(m, 0)))[0]
        ]



cdef class Analyzed:
    cdef public ndarray distances
    cdef public ndarray directions
    cdef public int is_reachable

    def __cinit__(self, ndarray distances, ndarray directions,
                   int is_reachable):
        self.distances = distances
        self.directions = directions
        self.is_reachable = is_reachable


    def path(self, row: int, column: int) -> List[Tuple[int, int]]:
        ret = []
        point = Point(row, column)
        if get_from_np_array(point, self.distances) == -1:
            raise ValueError(
                "No target point is reachable from: [{},{}]"
                    .format(point.x,point.y)
                )

        while True:
            symbol = get_from_np_array(point, self.directions)

            if symbol == b' ':
                return []

            ret.append(point.to_tuple())
            point = point.minus(SYMBOLS_TO_VECTORS[symbol])
            if symbol == b'X':
                break

        return ret

@cython.profile(False)
def flooding(start: List[Point], maze: ndarray) -> Analyzed:
    cdef list stack = [
        DistancePoint(
            s.x,
            s.y,
            0,
            s
            ) for s in start
        ]

    cdef ndarray distances = ndarray((maze.shape[0],maze.shape[1]),
                                      dtype='int')
    cdef ndarray directions = ndarray((maze.shape[0],maze.shape[1]),
                                   dtype='object')

    distances.fill(-1)
    directions.fill(b' ')

    for p in stack:
        p.set_distance(distances)
        p.set_direction(directions)

    cpdef DistancePoint new_point = None
    while (stack):
        current = stack.pop(0)
        for direction in neighbours:
            if can_go(current,maze,distances,direction):
                new_point = current.new_distance_point(
                    direction)
                new_point.set_distance(distances)
                new_point.set_direction(directions)
                stack.append(new_point)

    return Analyzed(
            distances,
            directions,
            is_reachable(distances)
        )


def is_reachable(dist: ndarray) -> bool:
    unique, counts = np.unique(dist, return_counts=True)
    d = dict(zip(unique, counts))
    return -1 not in d


cdef int is_in_bounds(Point p , ndarray a):
    return 0 <= p.x < a.shape[0] and 0 <= p.y < a.shape[1]

@cython.profile(False)
cdef int can_go(
        DistancePoint p,
        ndarray  maze,
        ndarray distances,
        Point dir
        ):
    cdef Point to = Point(p.x + dir.x, p.y + dir.y)

    if not is_in_bounds(to, maze) == 1:
        return False

    cdef int val = 0
    if dir == UP or dir == RIGHT:
        val = maze[to.x,to.y]
    else:
        val = maze[p.x,p.y]

    cdef int wall = is_nth_bit_set(val, DIRECTION_BITS[dir])
    cdef int distance = distances[to.x,to.y]
    return distance < 0 and not wall > 0


def get_from_np_array(p: Point, maze: ndarray):
    return maze[p.x,p.y]


def add(a: Point, b: Point) -> Point:
    return Point(a.x + b.x, a.y + b.y)


def minus(a: Point, b: Point) -> Point:
    return Point(a.x - b.x, a.y - b.y)


def analyze(maze: ndarray) -> Analyzed:
    if maze is None or maze.ndim != 2:
        raise TypeError("Input mazr must not be null and its dimension"
                        "must be exactly 2.")
    start = find_start(maze)
    return flooding(start, maze)
