"""
genetic/population.py – Genome model and evolution routines.
"""
from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config import (
    ELITE_COUNT,
    GENOME_SIZE,
    MUTATION_RATE,
    MUTATION_STD,
    POPULATION_SIZE,
    SELECTION_TOP,
)


@dataclass
class Genome:
    """Population member representing a neural network weight set."""

    weights: np.ndarray
    fitness: float = 0.0

    def __init__(self, weights: Optional[np.ndarray] = None) -> None:
        if weights is None:
            self.weights = np.random.randn(GENOME_SIZE).astype(np.float32) * 0.5
        else:
            self.weights = np.asarray(weights, dtype=np.float32)
        self.fitness = 0.0

    def clone(self) -> "Genome":
        return Genome(self.weights.copy())


def save_genome(genome: Genome, filename: str = "saved_genome.pkl") -> None:
    with open(filename, "wb") as handle:
        pickle.dump(genome, handle)


def load_best_genome(filename: str = "saved_genome.pkl") -> Optional[Genome]:
    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as handle:
        return pickle.load(handle)


def create_population_from_saved(seed: Genome) -> List[Genome]:
    population: List[Genome] = [seed.clone()]
    while len(population) < POPULATION_SIZE:
        child = seed.clone()
        perturbation = np.random.normal(0, 0.03, GENOME_SIZE).astype(np.float32)
        child.weights += perturbation
        population.append(child)
    return population


def _combine(parent_a: Genome, parent_b: Genome) -> Genome:
    mask = np.random.rand(GENOME_SIZE) < 0.5
    child = np.where(mask, parent_a.weights, parent_b.weights)
    return Genome(child)


def _perturb(genome: Genome) -> Genome:
    next_weights = genome.weights.copy()
    mutation_mask = np.random.rand(GENOME_SIZE) < MUTATION_RATE
    if mutation_mask.any():
        next_weights[mutation_mask] += np.random.randn(mutation_mask.sum()).astype(np.float32) * MUTATION_STD
    return Genome(next_weights)


def evolve(population: List[Genome]) -> List[Genome]:
    population.sort(key=lambda item: item.fitness, reverse=True)

    top_candidates = population[:SELECTION_TOP]
    save_genome(top_candidates[0], "best_car.pkl")

    next_gen = [candidate.clone() for candidate in top_candidates[:ELITE_COUNT]]
    while len(next_gen) < POPULATION_SIZE:
        parent_a, parent_b = random.sample(top_candidates, 2)
        offspring = _combine(parent_a, parent_b)
        next_gen.append(_perturb(offspring))

    return next_gen
