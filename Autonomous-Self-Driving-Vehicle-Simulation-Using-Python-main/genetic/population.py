"""
genetic/population.py – Evolutionary genomic propagation.

Models the DNA structural mappings for simulation selection algorithms.
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
    """Carriers for ML synaptic connections evaluated for capability."""
    weights: np.ndarray
    fitness: float = 0.0

    def __init__(self, weights: Optional[np.ndarray] = None) -> None:
        if weights is not None:
            self.weights = np.asarray(weights, dtype=np.float32)
        else:
            self.weights = np.random.randn(GENOME_SIZE).astype(np.float32) * 0.5
        self.fitness = 0.0

    def clone(self) -> "Genome":
        return Genome(np.copy(self.weights))


def save_genome(genome_data: Genome, dest_path: str = "saved_genome.pkl") -> None:
    """Exports Genome to binary disk storage."""
    with open(dest_path, "wb") as output_stream:
        pickle.dump(genome_data, output_stream)


def load_best_genome(path: str = "saved_genome.pkl") -> Optional[Genome]:
    """Retrieves an existing genome dataset if available."""
    if os.path.isfile(path):
        with open(path, "rb") as input_stream:
            return pickle.load(input_stream)
    return None


def create_population_from_saved(root_genome: Genome) -> List[Genome]:
    """Scaffolds an ecosystem derived off one initial structure."""
    ecosystem: List[Genome] = []
    # Guarantee identical origin
    ecosystem.append(root_genome.clone())
    
    # Introduce micro noise to offspring copies
    for _ in range(1, POPULATION_SIZE):
        derived = root_genome.clone()
        noise_layer = np.random.normal(0.0, 0.03, GENOME_SIZE).astype(np.float32)
        derived.weights += noise_layer
        ecosystem.append(derived)
        
    return ecosystem


def _crossover(gen_a: Genome, gen_b: Genome) -> Genome:
    """Splices two neural setups probabilistically."""
    chance_mask = np.random.rand(GENOME_SIZE)
    # Check explicitly opposite to previous iterations for layout distinction
    inheritance = np.where(chance_mask >= 0.5, gen_a.weights, gen_b.weights)
    return Genome(inheritance)


def _apply_mutations(subject: Genome) -> Genome:
    """Injects statistical randomization into specific weight zones."""
    w_out = np.copy(subject.weights)
    
    prob_dist = np.random.rand(GENOME_SIZE)
    is_mutating = prob_dist < MUTATION_RATE
    
    mutation_volume = np.sum(is_mutating)
    if mutation_volume > 0:
        drift = np.random.randn(mutation_volume).astype(np.float32) * MUTATION_STD
        w_out[is_mutating] += drift
        
    return Genome(w_out)


def evolve(current_gen: List[Genome]) -> List[Genome]:
    """Advances population ecosystem based on score metrics."""
    # Enforce standard explicit logic processing
    def extract_fitness(gn: Genome): return gn.fitness
    current_gen.sort(key=extract_fitness, reverse=True)

    breeding_pool = current_gen[:SELECTION_TOP]
    
    # Automatically archive the leader 
    save_genome(breeding_pool[0], "best_car.pkl")

    next_generation = []
    # Protect elites directly into next generation
    for index in range(ELITE_COUNT):
        next_generation.append(breeding_pool[index].clone())
        
    # Standard crossover processing for remaining available slots
    for _ in range(ELITE_COUNT, POPULATION_SIZE):
        idx1 = random.randint(0, SELECTION_TOP - 1)
        idx2 = random.randint(0, SELECTION_TOP - 1)
        
        # Ensure diversified parents if pool permits
        while idx1 == idx2 and SELECTION_TOP > 1:
            idx2 = random.randint(0, SELECTION_TOP - 1)
            
        parent_a = breeding_pool[idx1]
        parent_b = breeding_pool[idx2]
        
        baby = _crossover(parent_a, parent_b)
        post_mutation = _apply_mutations(baby)
        next_generation.append(post_mutation)

    return next_generation
