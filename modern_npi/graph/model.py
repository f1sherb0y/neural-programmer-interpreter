from modern_npi.graph.constants import ARGUMENT_DEPTHS, NUM_PROGRAMS
from modern_npi.graph.data import FEATURE_SIZE
from modern_npi.model import NeuralProgrammerInterpreter


class GraphNPI(NeuralProgrammerInterpreter):
    def __init__(self):
        super().__init__(
            feature_size=FEATURE_SIZE,
            num_programs=NUM_PROGRAMS,
            argument_depths=ARGUMENT_DEPTHS,
        )
