"""
car/car.py – Core vehicle structure and rendering.

Handles position tracking, orientation updates via neural inputs,
and Pygame drawing operations for the vehicle body and history.
"""

import math
from typing import List, Tuple

import numpy as np
import pygame
from config import (
    CAR_RADIUS,
    COL_BEST_CAR,
    COL_CAR_ALIVE,
    COL_CAR_DEAD,
    COL_SENSOR,
    COL_SENSOR_HIT,
    MAX_SPEED,
    MIN_SPEED,
    SENSOR_LENGTH,
    STEER_POWER,
)


class Car:
    def __init__(self, init_x: float, init_y: float, genome) -> None:
        self.location: pygame.Vector2 = pygame.Vector2(init_x, init_y)
        self.orientation: float = 0.0
        self.velocity: float = MIN_SPEED
        self.genome = genome
        
        self.is_active: bool = True
        self.distance_traveled: float = 0.0
        self.ticks_survived: int = 0
        self.accumulated_damage: float = 0.0
        
        # Initialize default sensor outputs
        self.sensor_data: List[float] = [float(SENSOR_LENGTH) for _ in range(7)]
        self.path_history: List[Tuple[int, int]] = []

    def execute_move(self, action_vector: np.ndarray) -> None:
        if not self.is_active:
            return

        # Extract commands mapping 0->steer, 1->throttle
        steering_input = float(action_vector[0])
        throttle_input = float(action_vector[1])

        # Kinematics update
        self.orientation += steering_input * STEER_POWER
        
        # Adjust velocity with bounds checking
        new_velocity = self.velocity + (throttle_input * 0.28)
        self.velocity = float(max(MIN_SPEED, min(new_velocity, MAX_SPEED)))

        rad = math.radians(self.orientation)
        
        # Vector displacement
        delta_x = math.cos(rad) * self.velocity
        delta_y = math.sin(rad) * self.velocity
        
        self.location.x += delta_x
        self.location.y -= delta_y

        # Progress tracking
        self.distance_traveled += self.velocity
        self.ticks_survived += 1
        
        self._record_path()

    def _record_path(self) -> None:
        # Save exact integral position to history map
        coord = (int(self.location.x), int(self.location.y))
        self.path_history.append(coord)
        # Shift out old data instead of growing infinitely
        while len(self.path_history) > 70:
            self.path_history.pop(0)

    def draw(self, target_surface: pygame.Surface, is_best: bool = False) -> None:
        if is_best:
            self._render_path(target_surface)

        self._render_chassis(target_surface, is_best)
        self._render_rays(target_surface, is_best)

    def _render_path(self, surface: pygame.Surface) -> None:
        history_size = len(self.path_history)
        if history_size < 2:
            return

        # Draw line segments with fading gradient based on iteration index
        for idx in range(1, history_size):
            p_prev = self.path_history[idx - 1]
            p_curr = self.path_history[idx]
            
            fade_factor = idx / history_size
            r_val = int(COL_BEST_CAR[0] * fade_factor)
            g_val = int(COL_BEST_CAR[1] * fade_factor)
            b_val = int(COL_BEST_CAR[2] * fade_factor * 0.6)
            
            pygame.draw.line(surface, (r_val, g_val, b_val), p_prev, p_curr, 2)

    def _render_chassis(self, surface: pygame.Surface, is_best: bool) -> None:
        # Determine heading normal vector
        rad_orient = math.radians(self.orientation)
        forward_x = math.cos(rad_orient)
        forward_y = -math.sin(rad_orient)
        
        heading_vec = pygame.Vector2(forward_x, forward_y)
        ortho_vec = pygame.Vector2(-forward_y, forward_x)

        # Map vertices
        v_front = self.location + (heading_vec * (CAR_RADIUS * 1.8))
        v_port = self.location + (ortho_vec * CAR_RADIUS)
        v_starboard = self.location - (ortho_vec * CAR_RADIUS)
        
        vertices = [v_front, v_port, v_starboard]

        if is_best:
            hull_paint = COL_BEST_CAR
            trim_paint = (220, 220, 220)
        else:
            hull_paint = COL_CAR_ALIVE if self.is_active else COL_CAR_DEAD
            trim_paint = (40, 40, 40)

        pygame.draw.polygon(surface, hull_paint, vertices)
        pygame.draw.polygon(surface, trim_paint, vertices, 1)

    def _render_rays(self, surface: pygame.Surface, is_best: bool) -> None:
        if not is_best:
            return

        num_sensors = len(self.sensor_data)
        spread = 120
        delta_angle = spread / (num_sensors - 1)

        # Plot all sensor hit marks and lines
        for idx in range(num_sensors):
            dist = self.sensor_data[idx]
            
            sweep_off = (spread / 2) - (idx * delta_angle)
            actual_angle = math.radians(self.orientation - sweep_off)
            
            target_px = self.location.x + (math.cos(actual_angle) * dist)
            target_py = self.location.y - (math.sin(actual_angle) * dist)
            hit_point = (target_px, target_py)
            
            # Use red hit color if object is near, else bright generic sensor color
            ray_clr = COL_SENSOR_HIT if dist < SENSOR_LENGTH else COL_SENSOR
            pygame.draw.line(surface, ray_clr, self.location, hit_point, 1)
            
            if dist < SENSOR_LENGTH:
                # Plop a circle right at collision coordinate
                pygame.draw.circle(surface, ray_clr, (int(target_px), int(target_py)), 3)
