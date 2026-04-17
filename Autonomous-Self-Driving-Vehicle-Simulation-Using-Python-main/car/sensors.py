"""
car/sensors.py – LiDAR/Raycasting obstacle detection mechanics.

Sweeps rays outward and calculates Euclidean bounds to
evaluate nearby collidable elements.
"""
import math
from typing import List

import pygame
from config import SENSOR_COUNT, SENSOR_LENGTH


class Sensors:
    def __init__(self) -> None:
        self.num_rays = SENSOR_COUNT
        self.viewing_angle = 120
        
        # Determine angle chunking to prevent division by zero gracefully
        if self.num_rays > 1:
            self.angle_increment = self.viewing_angle / (self.num_rays - 1)
        else:
            self.angle_increment = 0

    def get_readings(self, location: pygame.Vector2, orientation: float, track_mask: pygame.Mask) -> List[float]:
        mask_w, mask_h = track_mask.get_size()
        scan_results: List[float] = []

        # Iterate rays dynamically
        ray_idx = 0
        while ray_idx < self.num_rays:
            
            # Map out target degree based on starting heading and viewing angle spread
            sweep_offset = self.viewing_angle / 2.0
            target_deg = (orientation - sweep_offset) + (ray_idx * self.angle_increment)
            
            rads = math.radians(target_deg)
            delta_x = math.cos(rads)
            delta_y = -math.sin(rads)
            
            collision_distance = float(SENSOR_LENGTH)
            
            # Stepwise progression ray mapping
            step_depth = 1
            while step_depth <= SENSOR_LENGTH:
                pos_x = int(location.x + (delta_x * step_depth))
                pos_y = int(location.y + (delta_y * step_depth))
                
                # Check spatial matrix constraints
                x_invalid = (pos_x < 0) or (pos_x >= mask_w)
                y_invalid = (pos_y < 0) or (pos_y >= mask_h)
                
                if x_invalid or y_invalid or track_mask.get_at((pos_x, pos_y)):
                    collision_distance = float(step_depth)
                    break
                    
                step_depth += 1

            scan_results.append(collision_distance)
            ray_idx += 1

        return scan_results
