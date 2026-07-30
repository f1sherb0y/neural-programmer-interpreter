from enum import IntEnum


class Program(IntEnum):
    ADD = 0
    ADD1 = 1
    CARRY = 2
    LSHIFT = 3
    ACT = 4


class Action(IntEnum):
    MOVE = 0
    WRITE = 1
    DEFAULT = 2


class Pointer(IntEnum):
    INPUT1 = 0
    INPUT2 = 1
    CARRY = 2
    OUTPUT = 3
    DEFAULT = 4


class Direction(IntEnum):
    LEFT = 0
    RIGHT = 1


DEFAULT_ARGS = (Action.DEFAULT, Pointer.DEFAULT, 10)
ARG_DEPTHS = (3, 5, 11)
NUM_PROGRAMS = len(Program)
