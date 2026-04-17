"""
simulation/fitness.py – Fitness scoring algorithm.

Determines the effectiveness of genomic propagation based on
movement length, frame survival and obstacle collisions.
"""
from car.car import Car


def calculate_fitness(vehicle: Car) -> float:
    """Computes final evaluation metric for genetic prioritization."""
    
    # Progress acts as the primary evaluation anchor
    base_progress = vehicle.distance_traveled
    
    # Static weight variables
    duration_factor = 0.14
    structural_cost = 0.55
    
    # Dynamic calculations based on state attributes
    bonus_points = float(vehicle.ticks_survived) * duration_factor
    structural_penalty = float(vehicle.accumulated_damage) * structural_cost
    
    # Evaluate crash state deductibles
    inactive_fee = 20.0
    if vehicle.is_active:
        inactive_fee = 0.0
        
    net_score = (base_progress + bonus_points) - (structural_penalty + inactive_fee)

    return float(net_score)
