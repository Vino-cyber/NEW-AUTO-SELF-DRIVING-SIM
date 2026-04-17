"""
ai/neural_network.py – Controller inferencing architecture.

Maps serialized genome DNA into structural connection layers.
"""
from __future__ import annotations

import numpy as np
from config import HIDDEN_NODES, SENSOR_COUNT


class NeuralNetwork:
    """Feed-forward multi-layer perceptron for vehicular traversal calculations."""

    def __init__(self) -> None:
        self.node_inputs = SENSOR_COUNT
        self.node_hidden = HIDDEN_NODES
        self.node_outputs = 2

        # Dimension capacities
        self.cap_w1 = self.node_inputs * self.node_hidden
        self.cap_b1 = self.node_hidden
        self.cap_w2 = self.node_hidden * self.node_outputs
        self.cap_b2 = self.node_outputs

    def _extract_layers(self, dna_array: np.ndarray):
        """Consume sequence progressively via offset increments."""
        cursor = 0
        
        # Pull Input -> Hidden weights
        next_cursor = cursor + self.cap_w1
        w_in_hid = dna_array[cursor : next_cursor].reshape((self.node_inputs, self.node_hidden))
        cursor = next_cursor
        
        # Pull Hidden Bias
        next_cursor = cursor + self.cap_b1
        b_hid = dna_array[cursor : next_cursor]
        cursor = next_cursor
        
        # Pull Hidden -> Output weights
        next_cursor = cursor + self.cap_w2
        w_hid_out = dna_array[cursor : next_cursor].reshape((self.node_hidden, self.node_outputs))
        cursor = next_cursor
        
        # Pull Output Bias
        next_cursor = cursor + self.cap_b2
        b_out = dna_array[cursor : next_cursor]
        
        return w_in_hid, b_hid, w_hid_out, b_out

    def _propagate(self, data: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation."""
        return np.tanh(data)

    def infer(self, sensor_readings: np.ndarray, dna_matrix: np.ndarray) -> np.ndarray:
        """Calculate actuation commands over dual dense pathways."""
        arr_w1, arr_b1, arr_w2, arr_b2 = self._extract_layers(dna_matrix)
        
        # Dot product with active bounds restriction
        l1_feed = (sensor_readings @ arr_w1) + arr_b1
        l1_activation = self._propagate(l1_feed)
        
        l2_feed = (l1_activation @ arr_w2) + arr_b2
        final_emission = self._propagate(l2_feed)
        
        return final_emission
