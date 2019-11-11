import numpy as np
from typing import NamedTuple
from numpy import ndarray
from typing import Tuple, List

GO_LEFT_BIT = 1
GO_UP_BIT = 2


class Point(NamedTuple):
    x: int
    y: int

    def add(self, o):
        return add(self, o)

    def minus(self, o):
        return minus(self, o)

    def to_tuple(self):
        return (self.x,self.y)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class DistancePoint(Point):
    distance: int
    prev: Point

    def __new__(cls, x, y, distance, prev):
        self = super(DistancePoint, cls).__new__(cls, x, y)
        self.distance = distance
        self.prev = prev
        return self

    def get_point(self) -> Point:
        return Point(self.x, self.y)

    def set_distane(self, distances: ndarray):
        if get_from_np_array(self, distances) == -1:
            distances.itemset(
                (self.x, self.y),
                self.distance
                )

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def set_direction(self, directions: ndarray):
        if  get_from_np_array(self, directions) == b' ':
            directions.itemset(
                (self.x, self.y),
                DIRECTION_SYMBOLS[self.minus(self.prev)]
                )

    def new_distance_point(self, direction: Point):
        new = self.add(direction)
        return DistancePoint(
            x=new.x,
            y=new.y,
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

SYMBOLS_TO_VECTORS = { v:k for k,v in DIRECTION_SYMBOLS.items() }

neighbours = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def is_nth_bit_set(x: int, n: int) -> int:
    return x & (1 << n)


def find_start(m: ndarray) -> List[Point]:
    return [
        Point(t[0], t[1]) for t in np.dstack(np.where(is_nth_bit_set(m, 0)))[0]
        ]


class Analyzed(NamedTuple):
    distances: ndarray
    directions: ndarray
    is_reachable: bool

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


def flooding(start: List[Point], maze: ndarray) -> Analyzed:
    stack: List[DistancePoint] = [
        DistancePoint(
            s.x,
            s.y,
            0,
            s
            ) for s in start
        ]

    distances: ndarray = np.ndarray(maze.shape)
    directions: ndarray = np.ndarray(maze.shape, dtype='object')

    distances.fill(-1)
    directions.fill(b' ')

    for p in stack:
        p.set_distane(distances)
        p.set_direction(directions)

    while (stack):
        current = stack.pop(0)
        if can_go_up(current, maze, distances):
            new_point = current.new_distance_point(UP)
            new_point.set_distane(distances)
            new_point.set_direction(directions)
            stack.append(new_point)

        if can_go_down(current, maze, distances):
            new_point = current.new_distance_point(DOWN)
            new_point.set_distane(distances)
            new_point.set_direction(directions)
            stack.append(new_point)

        if can_go_left(current, maze, distances):
            new_point = current.new_distance_point(LEFT)
            new_point.set_distane(distances)
            new_point.set_direction(directions)
            stack.append(new_point)

        if can_go_right(current, maze, distances):
            new_point = current.new_distance_point( RIGHT)
            new_point.set_distane(distances)
            new_point.set_direction(directions)
            stack.append(new_point)

    return Analyzed(
        distances=distances,
        directions=directions,
        is_reachable=is_reachable(distances)
        )


def containst(p:Point, stack:List[Point]) -> bool:
    return filter(lambda x: p.__eq__(x),stack)[0]!=None

def is_reachable(dist: ndarray) -> bool:
    unique, counts = np.unique(dist, return_counts=True)
    d = dict(zip(unique, counts))
    return -1 not in d

def is_in_bounds(p: Point, a: ndarray) -> bool:
    return 0 <= p.x < a.shape[0] and 0 <= p.y < a.shape[1]


def can_go_left(p: Point, maze: ndarray, distances: ndarray) -> bool:
    to = p.add(LEFT)

    if not is_in_bounds(to, maze):
        return False

    val = get_from_np_array(p, maze)
    wall = is_nth_bit_set(val, GO_LEFT_BIT) > 0
    distance = get_from_np_array(to, distances)
    return distance < 0 and not wall


def can_go_right(p: Point, maze: ndarray, distances: ndarray) -> bool:
    to = p.add(RIGHT)

    if not is_in_bounds(to, maze):
        return False

    val = get_from_np_array(to, maze)
    wall = is_nth_bit_set(val, GO_LEFT_BIT) > 0
    distance = get_from_np_array(to, distances)
    return distance < 0 and not wall


def can_go_up(p: Point, maze: ndarray, distances: ndarray) -> bool:
    to = p.add(UP)

    if not is_in_bounds(to, maze):
        return False

    val = get_from_np_array(to, maze)
    wall = is_nth_bit_set(val, GO_UP_BIT) > 0
    distance = get_from_np_array(to, distances)
    return distance < 0 and not wall


def can_go_down(p: Point, maze: ndarray, distances: ndarray) -> bool:
    to = p.add(DOWN)

    if not is_in_bounds(to, maze):
        return False

    val = get_from_np_array(p, maze)
    wall = is_nth_bit_set(val, GO_UP_BIT) > 0
    distance = get_from_np_array(to, distances)
    return distance < 0 and not wall


def get_from_np_array(p: Point, maze: ndarray):
    return maze[p.x,p.y]


def add(a: Point, b: Point) -> Point:
    return Point(a.x + b.x, a.y + b.y)


def minus(a: Point, b: Point) -> Point:
    return Point(a.x - b.x, a.y - b.y)


def analyze(maze: ndarray) -> Analyzed:
    if maze == None or len(maze.shape) != 2:
        raise TypeError("Input mazr must not be null and its dimension"
                        "must be exactly 2.")
    start = find_start(maze)
    return flooding(start, maze)
