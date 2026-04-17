"""
car/sensors.py – Raycasting sensor model for the car.

Each sensor returns the nearest obstacle distance within a fixed range.
"""
import math
from typing import List

import pygame
from config import SENSOR_COUNT, SENSOR_LENGTH


class Sensors:
    def __init__(self) -> None:
        self.count = SENSOR_COUNT
        self.fov = 120
        self.step = self.fov / (self.count - 1) if self.count > 1 else 0

    def get_readings(self, position: pygame.Vector2, heading: float, track_mask: pygame.Mask) -> List[float]:
        width, height = track_mask.get_size()
        distances: List[float] = []

        for index in range(self.count):
            ray_angle = heading - self.fov / 2 + index * self.step
            radians = math.radians(ray_angle)
            dx = math.cos(radians)
            dy = -math.sin(radians)
            distance = float(SENSOR_LENGTH)

            for depth in range(1, SENSOR_LENGTH + 1):
                x = int(position.x + dx * depth)
                y = int(position.y + dy * depth)
                if x < 0 or x >= width or y < 0 or y >= height or track_mask.get_at((x, y)):
                    distance = float(depth)
                    break

            distances.append(distance)

        return distances
