import random

from npi.core.traces import Decision, Episode
from npi.tasks.addition.environment import AdditionEnvironment, decimal_add
from npi.tasks.addition.spec import (
    DEFAULT_ARGUMENTS,
    Action,
    Direction,
    Pointer,
    Program,
)


class AdditionTrace:
    def __init__(self, first: str, second: str):
        self.environment = AdditionEnvironment(first, second)
        self.episodes: list[Episode] = []
        self._run_add()
        expected = decimal_add(first, second)
        if self.environment.result() != expected:
            raise AssertionError(
                f"Reference trace produced {self.environment.result()}, expected {expected}"
            )

    def _episode(self, program: Program, arguments=DEFAULT_ARGUMENTS) -> Episode:
        episode = Episode(int(program), tuple(map(int, arguments)))
        self.episodes.append(episode)
        return episode

    def _decision(self, episode, child, arguments=DEFAULT_ARGUMENTS, *, end):
        episode.decisions.append(
            Decision(
                self.environment.observe(),
                end,
                int(child),
                tuple(map(int, arguments)),
                True,
            )
        )

    def _run_add(self):
        episode = self._episode(Program.ADD)
        while True:
            self._decision(episode, Program.ADD1, end=False)
            self._run_add1()
            finish = (
                self.environment.observe().at_most_significant_input == 1
                and self.environment.column_sum() < 10
            )
            self._decision(episode, Program.LSHIFT, end=finish)
            self._run_lshift()
            if finish:
                return

    def _run_add1(self):
        episode = self._episode(Program.ADD1)
        total = self.environment.column_sum()
        write = (Action.WRITE, Pointer.OUTPUT, total % 10)
        has_carry = total >= 10
        self._decision(episode, Program.ACT, write, end=not has_carry)
        self._run_act(write)
        if has_carry:
            self._decision(episode, Program.CARRY, end=True)
            self._run_carry()

    def _run_carry(self):
        episode = self._episode(Program.CARRY)
        actions = (
            (Action.MOVE, Pointer.CARRY, Direction.LEFT),
            (Action.WRITE, Pointer.CARRY, 1),
            (Action.MOVE, Pointer.CARRY, Direction.RIGHT),
        )
        for index, action in enumerate(actions):
            self._decision(episode, Program.ACT, action, end=index == 2)
            self._run_act(action)

    def _run_lshift(self):
        episode = self._episode(Program.LSHIFT)
        for pointer in range(4):
            action = (Action.MOVE, pointer, Direction.LEFT)
            self._decision(episode, Program.ACT, action, end=pointer == 3)
            self._run_act(action)

    def _run_act(self, arguments):
        episode = self._episode(Program.ACT, arguments)
        episode.decisions.append(
            Decision(
                self.environment.observe(),
                True,
                int(Program.ACT),
                DEFAULT_ARGUMENTS,
                False,
            )
        )
        self.environment.execute(tuple(map(int, arguments)))


def random_decimal(length: int, rng: random.Random) -> str:
    if length < 1:
        raise ValueError("Length must be positive")
    return str(rng.randint(1, 9)) + "".join(
        str(rng.randint(0, 9)) for _ in range(length - 1)
    )


def training_traces(
    examples_per_length: int = 32,
    maximum_length: int = 20,
    seed: int = 1,
) -> list[AdditionTrace]:
    rng = random.Random(seed)
    return [
        AdditionTrace(random_decimal(length, rng), random_decimal(length, rng))
        for length in range(1, maximum_length + 1)
        for _ in range(examples_per_length)
    ]
