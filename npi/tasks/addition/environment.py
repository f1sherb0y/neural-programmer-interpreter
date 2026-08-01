from dataclasses import dataclass

from npi.tasks.addition.spec import Action, Direction, Pointer


def digits_lsd_first(value: str) -> list[int]:
    if not value or not value.isdigit():
        raise ValueError(f"Expected an unsigned decimal string, got {value!r}")
    return [int(char) for char in reversed(value.lstrip("0") or "0")]


def decimal_add(first: str, second: str) -> str:
    first_digits = digits_lsd_first(first)
    second_digits = digits_lsd_first(second)
    carry = 0
    output = []
    for index in range(max(len(first_digits), len(second_digits))):
        total = carry
        if index < len(first_digits):
            total += first_digits[index]
        if index < len(second_digits):
            total += second_digits[index]
        output.append(total % 10)
        carry = total // 10
    if carry:
        output.append(carry)
    return "".join(str(digit) for digit in reversed(output))


@dataclass(frozen=True)
class AdditionObservation:
    digits: tuple[int, int, int, int]
    at_most_significant_input: int


class AdditionEnvironment:
    def __init__(self, first: str, second: str):
        first_digits = digits_lsd_first(first)
        second_digits = digits_lsd_first(second)
        self.input_width = max(len(first_digits), len(second_digits))
        capacity = self.input_width + 2
        self.rows = [[0] * capacity for _ in range(4)]
        self.rows[Pointer.INPUT1][: len(first_digits)] = first_digits
        self.rows[Pointer.INPUT2][: len(second_digits)] = second_digits
        self.pointers = [0, 0, 0, 0]

    def observe(self) -> AdditionObservation:
        digits = tuple(self.rows[row][self.pointers[row]] for row in range(4))
        return AdditionObservation(
            digits=digits,
            at_most_significant_input=int(
                self.pointers[Pointer.INPUT1] >= self.input_width - 1
            ),
        )

    def column_sum(self) -> int:
        return sum(self.observe().digits[:3])

    def execute(self, arguments: tuple[int, ...]) -> None:
        action, target, value = arguments
        if action == Action.MOVE:
            if target not in range(4):
                raise ValueError(f"Invalid pointer target: {target}")
            if value not in (Direction.LEFT, Direction.RIGHT):
                raise ValueError(f"Invalid pointer direction: {value}")
            delta = 1 if value == Direction.LEFT else -1
            next_position = self.pointers[target] + delta
            if not 0 <= next_position < len(self.rows[target]):
                raise IndexError(f"Pointer {target} moved outside the scratchpad")
            self.pointers[target] = next_position
            return
        if action == Action.WRITE:
            if target not in (Pointer.CARRY, Pointer.OUTPUT):
                raise ValueError(f"Cannot write through pointer {target}")
            if not 0 <= value <= 9:
                raise ValueError(f"Invalid decimal digit: {value}")
            self.rows[target][self.pointers[target]] = value
            return
        raise ValueError(f"Unknown action: {action}")

    def result(self) -> str:
        digits = self.rows[Pointer.OUTPUT]
        last = len(digits) - 1
        while last > 0 and digits[last] == 0:
            last -= 1
        return "".join(str(digit) for digit in reversed(digits[: last + 1]))
