from enum import IntEnum

from npi.core.spec import TaskSpec


class Program(IntEnum):
    DIJKSTRA = 0
    INITIALIZE = 1
    FIND_MIN = 2
    SETTLE = 3
    SCAN_EDGES = 4
    RELAX = 5
    ACT = 6


class Opcode(IntEnum):
    COPY_PTR = 0
    FOLLOW_PTR = 1
    READ_VAL = 2
    WRITE_PTR = 3
    WRITE_VAL = 4
    WRITE_BIT = 5
    ADD_VAL = 6
    CMP_VAL = 7


class Symbol(IntEnum):
    DEFAULT = 0
    FIRST = 1
    SOURCE = 2
    U = 3
    V = 4
    NODE = 5
    BEST = 6
    EDGE = 7
    NULL = 8
    D_U = 9
    D_V = 10
    D_BEST = 11
    WEIGHT = 12
    CANDIDATE = 13
    ZERO = 14
    INFINITY = 15
    NEXT_NODE = 16
    FIRST_EDGE = 17
    NEXT_EDGE = 18
    NEIGHBOR = 19
    PARENT = 20
    DISTANCE = 21
    EDGE_WEIGHT = 22
    SETTLED = 23
    BIT_ZERO = 24
    BIT_ONE = 25


POINTER_REGISTERS = {
    Symbol.FIRST,
    Symbol.SOURCE,
    Symbol.U,
    Symbol.V,
    Symbol.NODE,
    Symbol.BEST,
    Symbol.EDGE,
    Symbol.NULL,
}
WRITABLE_POINTER_REGISTERS = {
    Symbol.U,
    Symbol.V,
    Symbol.NODE,
    Symbol.BEST,
    Symbol.EDGE,
}
VALUE_REGISTERS = {
    Symbol.D_U,
    Symbol.D_V,
    Symbol.D_BEST,
    Symbol.WEIGHT,
    Symbol.CANDIDATE,
    Symbol.ZERO,
    Symbol.INFINITY,
}
WRITABLE_VALUE_REGISTERS = {
    Symbol.D_U,
    Symbol.D_V,
    Symbol.D_BEST,
    Symbol.WEIGHT,
    Symbol.CANDIDATE,
}
POINTER_FIELDS = {
    Symbol.NEXT_NODE,
    Symbol.FIRST_EDGE,
    Symbol.NEXT_EDGE,
    Symbol.NEIGHBOR,
    Symbol.PARENT,
}
VALUE_FIELDS = {Symbol.DISTANCE, Symbol.EDGE_WEIGHT}
BIT_FIELDS = {Symbol.SETTLED}
BIT_VALUES = {Symbol.BIT_ZERO, Symbol.BIT_ONE}
ARGUMENT_DEPTHS = (len(Opcode), len(Symbol), len(Symbol), len(Symbol))
DEFAULT_ARGUMENTS = (int(Opcode.COPY_PTR), 0, 0, 0)
SPEC = TaskSpec(
    name="weighted_dijkstra",
    observation_size=14,
    num_programs=len(Program),
    argument_depths=ARGUMENT_DEPTHS,
    action_program=int(Program.ACT),
    root_program=int(Program.DIJKSTRA),
    default_arguments=DEFAULT_ARGUMENTS,
    return_before_call=True,
)
