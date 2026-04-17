"""
ai/neural_network.py – Small feed-forward neural controller.

The network uses a flattened weight vector so the genetic algorithm can
apply crossover and mutation over a single continuous solution vector.
"""
from __future__ import annotations

import numpy as np
from config import HIDDEN_NODES, SENSOR_COUNT


class NeuralNetwork:
    """Minimal two-layer network for steering and throttle control."""

    def __init__(self) -> None:
        self.input_count = SENSOR_COUNT
        self.hidden_count = HIDDEN_NODES
        self.output_count = 2

        self._w1_size = self.input_count * self.hidden_count
        self._b1_size = self.hidden_count
        self._w2_size = self.hidden_count * self.output_count
        self._b2_size = self.output_count

        self._w1_end = self._w1_size
        self._b1_end = self._w1_end + self._b1_size
        self._w2_end = self._b1_end + self._w2_size
        self._b2_end = self._w2_end + self._b2_size

    def _unpack(self, genome: np.ndarray):
        w1 = genome[: self._w1_end].reshape(self.input_count, self.hidden_count)
        b1 = genome[self._w1_end : self._b1_end]
        w2 = genome[self._b1_end : self._w2_end].reshape(self.hidden_count, self.output_count)
        b2 = genome[self._w2_end : self._b2_end]
        return w1, b1, w2, b2

    def _activate(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def infer(self, sensors: np.ndarray, genome: np.ndarray) -> np.ndarray:
        """Produce steering and throttle outputs from sensor inputs."""
        w1, b1, w2, b2 = self._unpack(genome)
        hidden = self._activate(sensors @ w1 + b1)
        return self._activate(hidden @ w2 + b2)
