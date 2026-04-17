"""
car/car.py – Autonomous car physics, sensing, and rendering.

This car model tracks position, heading, speed, and neural genome state.
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
    def __init__(self, x: float, y: float, genome) -> None:
        self.pos: pygame.Vector2 = pygame.Vector2(x, y)
        self.angle: float = 0.0
        self.speed: float = MIN_SPEED
        self.genome = genome
        self.alive: bool = True
        self.distance: float = 0.0
        self.time_alive: int = 0
        self.damage: float = 0.0
        self.sensors: List[float] = [float(SENSOR_LENGTH)] * 7
        self.trail: List[Tuple[int, int]] = []

    def apply_controls(self, command: np.ndarray) -> None:
        if not self.alive:
            return

        steer = float(command[0])
        accel = float(command[1])

        self.angle += steer * STEER_POWER
        self.speed += accel * 0.28
        self.speed = float(np.clip(self.speed, MIN_SPEED, MAX_SPEED))

        angle_rad = math.radians(self.angle)
        self.pos.x += math.cos(angle_rad) * self.speed
        self.pos.y -= math.sin(angle_rad) * self.speed

        self.distance += self.speed
        self.time_alive += 1
        self._log_trail()

    def _log_trail(self) -> None:
        self.trail.append((int(self.pos.x), int(self.pos.y)))
        if len(self.trail) > 70:
            self.trail.pop(0)

    def draw(self, screen: pygame.Surface, highlight: bool = False) -> None:
        if highlight:
            self._draw_trail(screen)

        self._draw_vehicle(screen, highlight)
        self._draw_sensors(screen, highlight)

    def _draw_trail(self, screen: pygame.Surface) -> None:
        if len(self.trail) < 2:
            return

        for index in range(1, len(self.trail)):
            start = self.trail[index - 1]
            end = self.trail[index]
            ratio = index / len(self.trail)
            color = (
                int(COL_BEST_CAR[0] * ratio),
                int(COL_BEST_CAR[1] * ratio),
                int(COL_BEST_CAR[2] * ratio * 0.6),
            )
            pygame.draw.line(screen, color, start, end, 2)

    def _draw_vehicle(self, screen: pygame.Surface, highlight: bool) -> None:
        head = pygame.Vector2(math.cos(math.radians(self.angle)), -math.sin(math.radians(self.angle)))
        side = pygame.Vector2(-head.y, head.x)

        front = self.pos + head * (CAR_RADIUS * 1.8)
        left = self.pos + side * CAR_RADIUS
        right = self.pos - side * CAR_RADIUS

        body_color = COL_BEST_CAR if highlight else (COL_CAR_ALIVE if self.alive else COL_CAR_DEAD)
        pygame.draw.polygon(screen, body_color, [front, left, right])
        border = (220, 220, 220) if highlight else (40, 40, 40)
        pygame.draw.polygon(screen, border, [front, left, right], 1)

    def _draw_sensors(self, screen: pygame.Surface, highlight: bool) -> None:
        if not highlight:
            return

        count = len(self.sensors)
        fov = 120
        angle_step = fov / (count - 1)

        for index, distance in enumerate(self.sensors):
            ray_angle = self.angle - fov / 2 + index * angle_step
            ray_rad = math.radians(ray_angle)
            end = (
                self.pos.x + math.cos(ray_rad) * distance,
                self.pos.y - math.sin(ray_rad) * distance,
            )
            color = COL_SENSOR_HIT if distance < SENSOR_LENGTH else COL_SENSOR
            pygame.draw.line(screen, color, self.pos, end, 1)
            if distance < SENSOR_LENGTH:
                pygame.draw.circle(screen, color, (int(end[0]), int(end[1])), 3)
