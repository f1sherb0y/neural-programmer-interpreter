from dataclasses import dataclass

from npi.tasks.graph_ram.spec import (
    VALUE_REGISTERS,
    WRITABLE_REGISTERS,
    Opcode,
    Register,
)

NULL = -1
INFINITY = 10**12
HEADER_SOURCE = 0
HEADER_SCRATCH = 1
RECORD_SIZE = 3


@dataclass(frozen=True)
class RamObservation:
    comparison: tuple[int, int, int]

    def as_tuple(self):
        return self.comparison


class RamGraphEnvironment:
    def __init__(self, node_count, weighted_edges, source):
        if node_count < 1:
            raise ValueError("A graph must contain at least one node")
        if not 0 <= source < node_count:
            raise ValueError("Source is outside the graph")
        adjacency: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
        for start, end, weight in weighted_edges:
            if not (0 <= start < node_count and 0 <= end < node_count):
                raise ValueError("Edge endpoint is outside the graph")
            if weight < 0:
                raise ValueError("Dijkstra requires nonnegative weights")
            adjacency[start].append((end, weight))

        self.node_addresses = [2 + RECORD_SIZE * node for node in range(node_count)]
        self.memory = [0] * (2 + RECORD_SIZE * node_count)
        for address in self.node_addresses:
            self.memory[address] = NULL
            self.memory[address + 1] = INFINITY
            self.memory[address + 2] = NULL
        edge_count = 0
        for node, outgoing in enumerate(adjacency):
            previous = NULL
            for neighbor, weight in reversed(outgoing):
                address = len(self.memory)
                self.memory.extend((previous, self.node_addresses[neighbor], weight))
                previous = address
                edge_count += 1
            self.memory[self.node_addresses[node]] = previous

        self.memory[HEADER_SOURCE] = self.node_addresses[source]
        self.scratch_base = len(self.memory)
        self.memory[HEADER_SCRATCH] = self.scratch_base
        self.memory.extend([0] * (1 + 2 * (edge_count + 1)))
        self.registers = {register: 0 for register in Register}
        self.registers.update(
            {
                Register.DEFAULT: 0,
                Register.ZERO: 0,
                Register.ONE: 1,
                Register.TWO: 2,
                Register.NULL: NULL,
            }
        )
        self.comparison = (0, 0, 0)

    def observe(self):
        return RamObservation(self.comparison)

    def execute(self, action):
        opcode_value, first_value, second_value, third_value = action
        opcode = Opcode(opcode_value)
        first, second, third = map(Register, (first_value, second_value, third_value))
        if opcode == Opcode.MOV:
            self._writable(first)
            self._value(second)
            self._default(third)
            self.registers[first] = self.registers[second]
        elif opcode == Opcode.LOAD:
            self._writable(first)
            self._value(second)
            self._default(third)
            self.registers[first] = self.memory[self._address(second)]
        elif opcode == Opcode.STORE:
            self._value(first)
            self._value(second)
            self._default(third)
            self.memory[self._address(first)] = self.registers[second]
        elif opcode in (Opcode.ADD, Opcode.SUB):
            self._writable(first)
            self._value(second)
            self._value(third)
            if opcode == Opcode.ADD:
                self.registers[first] = self.registers[second] + self.registers[third]
            else:
                self.registers[first] = self.registers[second] - self.registers[third]
        elif opcode in (Opcode.SHL1, Opcode.SHR1):
            self._writable(first)
            self._value(second)
            self._default(third)
            if opcode == Opcode.SHL1:
                self.registers[first] = self.registers[second] * 2
            else:
                self.registers[first] = self.registers[second] // 2
        elif opcode == Opcode.CMP:
            self._value(first)
            self._value(second)
            self._default(third)
            left = self.registers[first]
            right = self.registers[second]
            self.comparison = (int(left < right), int(left == right), int(left > right))
        else:
            raise ValueError(f"Unsupported opcode: {opcode}")

    def distances(self):
        return [
            None if self.memory[address + 1] == INFINITY else self.memory[address + 1]
            for address in self.node_addresses
        ]

    def parents(self):
        address_to_node = {
            address: node for node, address in enumerate(self.node_addresses)
        }
        return [
            None
            if self.memory[address + 2] == NULL
            else address_to_node[self.memory[address + 2]]
            for address in self.node_addresses
        ]

    def result(self):
        return self.distances(), self.parents()

    def _address(self, register):
        address = self.registers[register]
        if not 0 <= address < len(self.memory):
            raise ValueError(f"Register {register.name} is not a valid memory address")
        return address

    @staticmethod
    def _writable(register):
        if register not in WRITABLE_REGISTERS:
            raise ValueError(f"Register {register.name} is not writable")

    @staticmethod
    def _value(register):
        if register not in VALUE_REGISTERS:
            raise ValueError(f"Register {register.name} has no readable value")

    @staticmethod
    def _default(register):
        if register != Register.DEFAULT:
            raise ValueError("Unused argument must be DEFAULT")
