"""
genetic/selection.py – Ranked selection utility.
"""
from typing import List

from config import SELECTION_TOP
from genetic.population import Genome


def select_top(population: List[Genome]) -> List[Genome]:
    """Return the highest-fitness genomes from the population."""
    sorted_population = sorted(population, key=lambda genome: genome.fitness, reverse=True)
    return sorted_population[:SELECTION_TOP]
