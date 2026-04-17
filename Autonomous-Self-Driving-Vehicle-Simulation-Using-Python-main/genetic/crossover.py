"""
genetic/crossover.py – Mix parent genomes.
"""
import numpy as np

from config import GENOME_SIZE
from genetic.population import Genome


def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
    """Combine two parent genomes into a new child genome."""
    mask = np.random.rand(GENOME_SIZE) < 0.5
    child_weights = np.where(mask, parent_a.weights, parent_b.weights)
    return Genome(child_weights)
