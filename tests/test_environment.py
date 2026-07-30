import unittest

from modern_npi.constants import Program
from modern_npi.environment import AdditionEnvironment, decimal_add
from modern_npi.traces import AdditionTrace


class AdditionEnvironmentTest(unittest.TestCase):
    def test_reference_trace_handles_cascading_carry(self):
        trace = AdditionTrace("999", "1")
        self.assertEqual(trace.environment.result(), "1000")

    def test_reference_trace_is_hierarchical(self):
        trace = AdditionTrace("96", "125")
        programs = [episode.program for episode in trace.episodes]
        self.assertEqual(programs.count(Program.ADD), 1)
        self.assertEqual(programs.count(Program.ADD1), 3)
        self.assertEqual(programs.count(Program.LSHIFT), 3)
        self.assertEqual(len(trace.episodes[0].decisions), 6)

    def test_leading_zero_input_is_normalized(self):
        trace = AdditionTrace("0009", "0001")
        self.assertEqual(trace.environment.result(), "10")

    def test_decimal_add_is_not_limited_by_python_integer_conversion(self):
        first = "9" * 10_000
        self.assertEqual(decimal_add(first, "1"), "1" + "0" * 10_000)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            AdditionEnvironment("-1", "2")


if __name__ == "__main__":
    unittest.main()
