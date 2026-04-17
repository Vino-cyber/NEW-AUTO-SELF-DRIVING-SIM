"""
genetic/mutation.py – Add noise to genome weights.
"""
import numpy as np

from config import GENOME_SIZE, MUTATION_RATE, MUTATION_STD
from genetic.population import Genome


def mutate(genome: Genome) -> Genome:
    """Mutate a genome by randomly perturbing selected weights."""
    weights = genome.weights.copy()
    mask = np.random.rand(GENOME_SIZE) < MUTATION_RATE
    if mask.any():
        weights[mask] += np.random.randn(mask.sum()).astype(np.float32) * MUTATION_STD
    return Genome(weights)
