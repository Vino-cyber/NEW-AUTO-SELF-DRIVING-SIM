"""
simulation/fitness.py – Compute a performance score for each car.

Fitness is based on progress, endurance, and damage control.
"""
from car.car import Car


def calculate_fitness(car: Car) -> float:
    """Return a scalar reward for the car's run performance."""
    progress_score = car.distance
    survival_bonus = car.time_alive * 0.14
    damage_cost = car.damage * 0.55
    status_penalty = 0.0 if car.alive else 20.0

    return progress_score + survival_bonus - damage_cost - status_penalty
