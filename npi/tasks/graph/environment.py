from dataclasses import dataclass

from npi.tasks.graph.spec import (
    BIT_FIELDS,
    BIT_VALUES,
    POINTER_FIELDS,
    POINTER_REGISTERS,
    VALUE_FIELDS,
    VALUE_REGISTERS,
    WRITABLE_POINTER_REGISTERS,
    WRITABLE_VALUE_REGISTERS,
    Opcode,
    Symbol,
)

NULL = -1
INFINITY = 10**12


@dataclass
class NodeRecord:
    next_node: int
    first_edge: int = NULL
    distance: int = INFINITY
    parent: int = NULL
    settled: bool = False


@dataclass
class EdgeRecord:
    next_edge: int
    neighbor: int
    weight: int


@dataclass(frozen=True)
class GraphObservation:
    pointer_null: tuple[int, ...]
    settled: tuple[int, ...]
    comparison: tuple[int, int, int]

    def as_tuple(self) -> tuple[int, ...]:
        return self.pointer_null + self.settled + self.comparison


class GraphEnvironment:
    pointer_observation_registers = (
        Symbol.FIRST,
        Symbol.SOURCE,
        Symbol.U,
        Symbol.V,
        Symbol.NODE,
        Symbol.BEST,
        Symbol.EDGE,
    )
    settled_observation_registers = (
        Symbol.U,
        Symbol.V,
        Symbol.NODE,
        Symbol.BEST,
    )

    def __init__(
        self,
        node_count: int,
        weighted_edges: list[tuple[int, int, int]],
        source: int,
    ):
        if node_count < 1:
            raise ValueError("A graph must contain at least one node")
        if not 0 <= source < node_count:
            raise ValueError("Source is outside the graph")
        self.nodes = [
            NodeRecord(next_node=index + 1 if index + 1 < node_count else NULL)
            for index in range(node_count)
        ]
        adjacency: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
        for start, end, weight in weighted_edges:
            if not (0 <= start < node_count and 0 <= end < node_count):
                raise ValueError("Edge endpoint is outside the graph")
            if weight < 0:
                raise ValueError("Dijkstra requires nonnegative weights")
            adjacency[start].append((end, weight))

        self.edges: list[EdgeRecord] = []
        for node, outgoing in enumerate(adjacency):
            previous = NULL
            for neighbor, weight in reversed(outgoing):
                edge_index = len(self.edges)
                self.edges.append(EdgeRecord(previous, neighbor, weight))
                previous = edge_index
            self.nodes[node].first_edge = previous

        self.pointer_registers = {
            Symbol.FIRST: 0,
            Symbol.SOURCE: source,
            Symbol.U: NULL,
            Symbol.V: NULL,
            Symbol.NODE: NULL,
            Symbol.BEST: NULL,
            Symbol.EDGE: NULL,
            Symbol.NULL: NULL,
        }
        self.value_registers = {
            Symbol.D_U: 0,
            Symbol.D_V: 0,
            Symbol.D_BEST: 0,
            Symbol.WEIGHT: 0,
            Symbol.CANDIDATE: 0,
            Symbol.ZERO: 0,
            Symbol.INFINITY: INFINITY,
        }
        self.comparison = (0, 0, 0)

    def observe(self) -> GraphObservation:
        pointer_null = tuple(
            int(self.pointer_registers[register] == NULL)
            for register in self.pointer_observation_registers
        )
        settled = tuple(
            int(self._node_for_register(register).settled)
            if self.pointer_registers[register] != NULL
            else 0
            for register in self.settled_observation_registers
        )
        return GraphObservation(pointer_null, settled, self.comparison)

    def execute(self, action: tuple[int, int, int, int]) -> None:
        opcode_value, first_value, second_value, third_value = action
        opcode = Opcode(opcode_value)
        first = Symbol(first_value)
        second = Symbol(second_value)
        third = Symbol(third_value)

        if opcode == Opcode.COPY_PTR:
            self._require(first, WRITABLE_POINTER_REGISTERS, "pointer destination")
            self._require(second, POINTER_REGISTERS, "pointer source")
            self._require_default(third)
            self.pointer_registers[first] = self.pointer_registers[second]
            return

        if opcode == Opcode.FOLLOW_PTR:
            self._require(first, WRITABLE_POINTER_REGISTERS, "pointer destination")
            self._require(second, POINTER_REGISTERS, "object pointer")
            self._require(third, POINTER_FIELDS, "pointer field")
            self.pointer_registers[first] = self._read_pointer_field(second, third)
            return

        if opcode == Opcode.READ_VAL:
            self._require(first, WRITABLE_VALUE_REGISTERS, "value destination")
            self._require(second, POINTER_REGISTERS, "object pointer")
            self._require(third, VALUE_FIELDS, "value field")
            self.value_registers[first] = self._read_value_field(second, third)
            return

        if opcode == Opcode.WRITE_PTR:
            self._require(first, POINTER_REGISTERS, "object pointer")
            self._require(second, POINTER_FIELDS, "pointer field")
            self._require(third, POINTER_REGISTERS, "pointer source")
            if second != Symbol.PARENT:
                raise ValueError("Only parent pointers are writable")
            self._node_for_register(first).parent = self.pointer_registers[third]
            return

        if opcode == Opcode.WRITE_VAL:
            self._require(first, POINTER_REGISTERS, "object pointer")
            self._require(second, VALUE_FIELDS, "value field")
            self._require(third, VALUE_REGISTERS, "value source")
            if second != Symbol.DISTANCE:
                raise ValueError("Only node distances are writable")
            self._node_for_register(first).distance = self.value_registers[third]
            return

        if opcode == Opcode.WRITE_BIT:
            self._require(first, POINTER_REGISTERS, "object pointer")
            self._require(second, BIT_FIELDS, "bit field")
            self._require(third, BIT_VALUES, "bit value")
            self._node_for_register(first).settled = third == Symbol.BIT_ONE
            return

        if opcode == Opcode.ADD_VAL:
            self._require(first, WRITABLE_VALUE_REGISTERS, "value destination")
            self._require(second, VALUE_REGISTERS, "left value")
            self._require(third, VALUE_REGISTERS, "right value")
            self.value_registers[first] = (
                self.value_registers[second] + self.value_registers[third]
            )
            return

        if opcode == Opcode.CMP_VAL:
            self._require(first, VALUE_REGISTERS, "left value")
            self._require(second, VALUE_REGISTERS, "right value")
            self._require_default(third)
            left = self.value_registers[first]
            right = self.value_registers[second]
            self.comparison = (int(left < right), int(left == right), int(left > right))
            return

        raise ValueError(f"Unsupported opcode: {opcode}")

    def distances(self) -> list[int | None]:
        return [
            None if node.distance == INFINITY else node.distance for node in self.nodes
        ]

    def parents(self) -> list[int | None]:
        return [None if node.parent == NULL else node.parent for node in self.nodes]

    def result(self) -> tuple[list[int | None], list[int | None]]:
        return self.distances(), self.parents()

    def _node_for_register(self, register: Symbol) -> NodeRecord:
        pointer = self.pointer_registers[register]
        if not 0 <= pointer < len(self.nodes):
            raise ValueError(f"Register {register.name} does not point to a node")
        return self.nodes[pointer]

    def _edge_for_register(self, register: Symbol) -> EdgeRecord:
        pointer = self.pointer_registers[register]
        if not 0 <= pointer < len(self.edges):
            raise ValueError(f"Register {register.name} does not point to an edge")
        return self.edges[pointer]

    def _read_pointer_field(self, object_register: Symbol, field: Symbol) -> int:
        if field == Symbol.NEXT_NODE:
            return self._node_for_register(object_register).next_node
        if field == Symbol.FIRST_EDGE:
            return self._node_for_register(object_register).first_edge
        if field == Symbol.NEXT_EDGE:
            return self._edge_for_register(object_register).next_edge
        if field == Symbol.NEIGHBOR:
            return self._edge_for_register(object_register).neighbor
        if field == Symbol.PARENT:
            return self._node_for_register(object_register).parent
        raise ValueError(f"Unsupported pointer field: {field}")

    def _read_value_field(self, object_register: Symbol, field: Symbol) -> int:
        if field == Symbol.DISTANCE:
            return self._node_for_register(object_register).distance
        if field == Symbol.EDGE_WEIGHT:
            return self._edge_for_register(object_register).weight
        raise ValueError(f"Unsupported value field: {field}")

    @staticmethod
    def _require(value: Symbol, allowed: set[Symbol], role: str) -> None:
        if value not in allowed:
            raise ValueError(f"{value.name} is not a valid {role}")

    @staticmethod
    def _require_default(value: Symbol) -> None:
        if value != Symbol.DEFAULT:
            raise ValueError("Unused argument must be DEFAULT")
