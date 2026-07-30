import torch
from torch import nn

from modern_npi.constants import ARG_DEPTHS, NUM_PROGRAMS
from modern_npi.data import FEATURE_SIZE


class NeuralProgrammerInterpreter(nn.Module):
    def __init__(
        self,
        *,
        state_size: int = 128,
        program_size: int = 64,
        key_size: int = 32,
        hidden_size: int = 256,
        layers_count: int = 2,
        feature_size: int = FEATURE_SIZE,
        num_programs: int = NUM_PROGRAMS,
        argument_depths: tuple[int, ...] = ARG_DEPTHS,
    ):
        super().__init__()
        self.state_size = state_size
        self.program_size = program_size
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.layers_count = layers_count
        self.feature_size = feature_size
        self.num_programs = num_programs
        self.argument_depths = argument_depths

        self.state_encoder = nn.Sequential(
            nn.Linear(feature_size, state_size),
            nn.ReLU(),
            nn.Linear(state_size, state_size),
        )
        self.program_embeddings = nn.Embedding(num_programs, program_size)
        self.program_keys = nn.Parameter(torch.empty(num_programs, key_size))
        nn.init.xavier_uniform_(self.program_keys)
        self.recurrent_core = nn.LSTM(
            state_size + program_size,
            hidden_size,
            num_layers=layers_count,
            batch_first=True,
        )
        self.step_cells = nn.ModuleList(
            [
                nn.LSTMCell(
                    state_size + program_size if index == 0 else hidden_size,
                    hidden_size,
                )
                for index in range(layers_count)
            ]
        )
        self.end_head = nn.Linear(hidden_size, 2)
        self.key_head = nn.Linear(hidden_size, key_size)
        self.argument_heads = nn.ModuleList(
            [nn.Linear(hidden_size, depth) for depth in argument_depths]
        )
        self._tie_step_cells()

    def _tie_step_cells(self) -> None:
        """Share the sequence LSTM parameters with the one-step LSTMCells."""
        for index, cell in enumerate(self.step_cells):
            cell.weight_ih = getattr(self.recurrent_core, f"weight_ih_l{index}")
            cell.weight_hh = getattr(self.recurrent_core, f"weight_hh_l{index}")
            cell.bias_ih = getattr(self.recurrent_core, f"bias_ih_l{index}")
            cell.bias_hh = getattr(self.recurrent_core, f"bias_hh_l{index}")

    def _decode(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        end_logits = self.end_head(hidden)
        predicted_key = self.key_head(hidden)
        program_logits = predicted_key @ self.program_keys.T
        argument_logits = tuple(head(hidden) for head in self.argument_heads)
        return end_logits, program_logits, argument_logits

    def forward(
        self,
        features: torch.Tensor,
        program_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        state = self.state_encoder(features)
        program = self.program_embeddings(program_ids)
        hidden, _ = self.recurrent_core(torch.cat((state, program), dim=-1))
        return self._decode(hidden)

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device),
            )
            for _ in range(self.layers_count)
        ]

    def inference_step(
        self,
        features: torch.Tensor,
        program_ids: torch.Tensor,
        states: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        current = torch.cat(
            (self.state_encoder(features), self.program_embeddings(program_ids)), dim=-1
        )
        next_states = []
        for cell, state in zip(self.step_cells, states, strict=True):
            hidden, memory = cell(current, state)
            next_states.append((hidden, memory))
            current = hidden
        end_logits, program_logits, argument_logits = self._decode(current)
        return end_logits, program_logits, argument_logits, next_states
