from enum import IntEnum

from npi.core.spec import TaskSpec


class Program(IntEnum):
    DIJKSTRA = 0
    INITIALIZE = 1
    INSERT = 2
    BUBBLE_UP = 3
    EXTRACT = 4
    BUBBLE_DOWN = 5
    SCAN_EDGES = 6
    RELAX = 7
    ACT = 8


class Opcode(IntEnum):
    MOV = 0
    LOAD = 1
    STORE = 2
    ADD = 3
    SUB = 4
    SHL1 = 5
    SHR1 = 6
    CMP = 7


class Register(IntEnum):
    DEFAULT = 0
    ZERO = 1
    ONE = 2
    TWO = 3
    NULL = 4
    R0 = 5
    R1 = 6
    R2 = 7
    R3 = 8
    R4 = 9
    R5 = 10
    R6 = 11
    R7 = 12
    R8 = 13
    R9 = 14
    R10 = 15
    R11 = 16
    R12 = 17
    R13 = 18
    R14 = 19
    R15 = 20
    R16 = 21
    R17 = 22
    R18 = 23
    R19 = 24
    R20 = 25
    R21 = 26
    R22 = 27
    R23 = 28


READ_ONLY_REGISTERS = {
    Register.DEFAULT,
    Register.ZERO,
    Register.ONE,
    Register.TWO,
    Register.NULL,
}
WRITABLE_REGISTERS = set(Register) - READ_ONLY_REGISTERS
VALUE_REGISTERS = set(Register) - {Register.DEFAULT}
ARGUMENT_DEPTHS = (len(Opcode), len(Register), len(Register), len(Register))
DEFAULT_ARGUMENTS = (int(Opcode.MOV), 0, 0, 0)
SPEC = TaskSpec(
    name="ram_dijkstra",
    observation_size=3,
    num_programs=len(Program),
    argument_depths=ARGUMENT_DEPTHS,
    action_program=int(Program.ACT),
    root_program=int(Program.DIJKSTRA),
    default_arguments=DEFAULT_ARGUMENTS,
    return_before_call=True,
)
