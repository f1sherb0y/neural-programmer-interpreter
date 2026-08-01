from enum import IntEnum

from npi.core.spec import TaskSpec


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


ARGUMENT_DEPTHS = (3, 5, 11)
DEFAULT_ARGUMENTS = (int(Action.DEFAULT), int(Pointer.DEFAULT), 10)
SPEC = TaskSpec(
    name="addition",
    observation_size=41,
    num_programs=len(Program),
    argument_depths=ARGUMENT_DEPTHS,
    action_program=int(Program.ACT),
    root_program=int(Program.ADD),
    default_arguments=DEFAULT_ARGUMENTS,
    return_before_call=False,
)
