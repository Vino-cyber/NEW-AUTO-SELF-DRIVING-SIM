"""
Track builder for the driving simulation.

This module lays out the road surface, adds visual track details,
and builds a mask for off-track collision detection.
"""
import math
import random
from typing import List, Tuple
import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT

random.seed(13)

# Basic color palette
GRASS = (58, 105, 42)
GRAVEL = (150, 133, 100)
ASPHALT = (58, 55, 50)
YELLOW = (208, 182, 48)
WHITE = (202, 197, 180)
KRED = (188, 40, 30)  # Kerb red
KWHT = (222, 218, 206)  # Kerb white
DARK = (30, 28, 24)

# Road geometry constants
ROAD_HALF_WIDTH = 62  # Road half-width → full stroke width = 124 px
SHOULDER_HALF_WIDTH = ROAD_HALF_WIDTH + 13  # Full stroke = 150 px

# Legacy variable aliases
RW = ROAD_HALF_WIDTH
SW = SHOULDER_HALF_WIDTH

# Starting car position
SPAWN_X = 220
SPAWN_Y = 500  # well away from the start-line markings at y≈400
SPAWN_ANGLE = 90.0  # pointing upward along left straight

# Geometry construction constants
CORNER_RADIUS = 90
LEFT_STRAIGHT_X = 220
RIGHT_STRAIGHT_X = 820
TOP_STRAIGHT_Y = 120
BOTTOM_STRAIGHT_Y = 680

TOP_LEFT_CENTER = (310, 210)
TOP_RIGHT_CENTER = (730, 210)
BOTTOM_RIGHT_CENTER = (730, 590)
BOTTOM_LEFT_CENTER = (310, 590)

# Chicane layout values
CHICANE_START_X = 480
CHICANE_PEAK_X = 530
CHICANE_PEAK_OFFSET = 45
CHICANE_END_X = 580

# Extra visual detail settings
GRAIN_SAMPLES = 6000
GRAIN_VARIATION = 7
GRAIN_Y_BOUNDS = (90, SCREEN_HEIGHT - 80)
GRAIN_X_BOUNDS = (130, SCREEN_WIDTH - 230)

DASH_PROBABILITY = 0.18
DASH_SPACING_BASE = 8
DASH_RANDOM_GAP = (12, 32)

EDGE_PROBABILITY = 0.28
EDGE_SEGMENT_LENGTH = 5
EDGE_RANDOM_GAP = (7, 20)

KERB_SECTIONS = [0.04, 0.28, 0.53, 0.78]
KERB_LENGTH = 24

POTHOLE_TARGET_COUNT = 10
POTHOLE_MAX_ATTEMPTS = 500
POTHOLE_DISTANCE_RANGE = (15, 45)
POTHOLE_SIZE_RANGE = ((5, 11), (4, 7))  # (width, height)

BUMP_SECTIONS = [0.10, 0.36, 0.56, 0.74, 0.90]
BUMP_STRIPE_COUNT = 9

OBSTACLE_CONFIG = [
    (0.22, 46, 20, 11, (255, 195, 0)),      # Moved from 0.15 away from bump at 0.10
    (0.44, -48, 18, 10, (148, 50, 30)),     # Moved from 0.40 away from bump at 0.36
    (0.65, 45, 15, 9, (70, 70, 70)),        # Moved from 0.60 away from bump at 0.56
    (0.85, -46, 22, 11, (208, 190, 150)),   # Moved from 0.78 away from kerb at 0.78
]

STARTLINE_TILE_SIZE = 11
STARTLINE_TILE_COUNT = 10


# Helper functions
def _arc_points(
    center_x: float, center_y: float, radius: float,
    angle_start: float, angle_end: float, segments: int = 40
) -> List[Tuple[float, float]]:
    """Create evenly spaced points along a circular arc."""
    points: List[Tuple[float, float]] = []
    for i in range(segments + 1):
        angle_deg = angle_start + (angle_end - angle_start) * i / segments
        angle_rad = math.radians(angle_deg)
        x = center_x + math.cos(angle_rad) * radius
        y = center_y + math.sin(angle_rad) * radius
        points.append((x, y))
    return points


def _linear_points(
    x0: float, y0: float, x1: float, y1: float, step: int = 6
) -> List[Tuple[float, float]]:
    """Sample points with a uniform spacing along a straight segment."""
    distance = math.hypot(x1 - x0, y1 - y0)
    num_points = max(2, int(distance / step))
    return [
        (x0 + (x1 - x0) * i / (num_points - 1),
         y0 + (y1 - y0) * i / (num_points - 1))
        for i in range(num_points)
    ]


def _build_center_line() -> List[Tuple[float, float]]:
    """
    Build the race track centerline as an ordered sequence of points.

    Layout (clockwise, y increases downward):
    - Left straight segment
    - Top straight segment with chicane
    - Right straight segment
    - Bottom straight segment
    """
    center_line: List[Tuple[float, float]] = []

    center_line += _linear_points(LEFT_STRAIGHT_X, BOTTOM_LEFT_CENTER[1],
                                  LEFT_STRAIGHT_X, TOP_LEFT_CENTER[1])
    center_line += _arc_points(*TOP_LEFT_CENTER, CORNER_RADIUS, 180, 270)
    center_line += _linear_points(TOP_LEFT_CENTER[0], TOP_STRAIGHT_Y,
                                  CHICANE_START_X, TOP_STRAIGHT_Y)
    center_line += _linear_points(CHICANE_START_X, TOP_STRAIGHT_Y,
                                  CHICANE_PEAK_X, TOP_STRAIGHT_Y + CHICANE_PEAK_OFFSET)
    center_line += _linear_points(CHICANE_PEAK_X, TOP_STRAIGHT_Y + CHICANE_PEAK_OFFSET,
                                  CHICANE_END_X, TOP_STRAIGHT_Y)
    center_line += _linear_points(CHICANE_END_X, TOP_STRAIGHT_Y,
                                  TOP_RIGHT_CENTER[0], TOP_STRAIGHT_Y)
    center_line += _arc_points(*TOP_RIGHT_CENTER, CORNER_RADIUS, 270, 360)
    center_line += _linear_points(RIGHT_STRAIGHT_X, TOP_RIGHT_CENTER[1],
                                  RIGHT_STRAIGHT_X, BOTTOM_RIGHT_CENTER[1])
    center_line += _arc_points(*BOTTOM_RIGHT_CENTER, CORNER_RADIUS, 0, 90)
    center_line += _linear_points(BOTTOM_RIGHT_CENTER[0], BOTTOM_STRAIGHT_Y,
                                  BOTTOM_LEFT_CENTER[0], BOTTOM_STRAIGHT_Y)
    center_line += _arc_points(*BOTTOM_LEFT_CENTER, CORNER_RADIUS, 90, 180)

    return center_line

# Draw a thick path by connecting points with circles and lines
def _render_thick_path(surface, points, color, width):
    radius = width // 2
    integer_points = [(int(x), int(y)) for x, y in points]
    for i in range(len(integer_points) - 1):
        pygame.draw.line(surface, color, integer_points[i], integer_points[i + 1], width)
        pygame.draw.circle(surface, color, integer_points[i], radius)
    pygame.draw.circle(surface, color, integer_points[-1], radius)

# Track class and rendering utilities
class Track:
    """Represents the race track with road surface, obstacles, and collision detection."""
    
    def __init__(self):
        self.surface: pygame.Surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.obstacle_rects: List[pygame.Rect] = []
        self.pothole_rects: List[pygame.Rect] = []  # Used when checking for car damage
        self.speedbreaker_rects: List[pygame.Rect] = []  # Used when checking speed bump collisions
        self._spine: List[Tuple[float, float]] = _build_center_line()
        self._build()
        
        # Build a mask where grass pixels count as off-track
        self.mask = pygame.mask.from_threshold(self.surface, GRASS, (20, 20, 20))
        
        # Draw the start/finish line after the mask is ready
        self._draw_startline(self.surface, self._spine)

    def _build(self) -> None:
        """Build the complete track surface with all visual elements."""
        surface, spine = self.surface, self._spine
        surface.fill(GRASS)
        
        _render_thick_path(surface, spine, GRAVEL, SW * 2)
        _render_thick_path(surface, spine, ASPHALT, RW * 2)
        
        self._add_grain(surface)
        self._add_dashes(surface, spine)
        self._add_edges(surface, spine)
        self._add_kerbs(surface, spine)
        self._add_potholes(surface, spine)
        self._add_bumps(surface, spine)
        self._add_obstacles(surface, spine)

    def _get_normal_at_index(self, spine_index: int) -> Tuple[float, float]:
        """
        Calculate the perpendicular (normal) vector to the spine at a given index.
        Returns a unit normal vector (nx, ny).
        """
        spine = self._spine
        num_points = len(spine)
        
        p0 = spine[max(0, spine_index - 1)]
        p1 = spine[min(num_points - 1, spine_index + 1)]
        
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = math.hypot(dx, dy) or 1
        
        # Return perpendicular (rotated 90° counterclockwise)
        return -dy / length, dx / length

    def _add_grain(self, surface: pygame.Surface) -> None:
        """Add random noise texture to asphalt for visual realism."""
        for _ in range(GRAIN_SAMPLES):
            x = random.randint(*GRAIN_X_BOUNDS)
            y = random.randint(*GRAIN_Y_BOUNDS)
            
            try:
                pixel_color = surface.get_at((x, y))[:3]
                if pixel_color == ASPHALT:
                    variation = random.randint(-GRAIN_VARIATION, GRAIN_VARIATION)
                    new_color = tuple(
                        max(0, min(255, c + variation)) 
                        for c in ASPHALT
                    )
                    surface.set_at((x, y), new_color)
            except IndexError:
                pass

    def _add_dashes(self, surface: pygame.Surface, spine: List[Tuple[float, float]]) -> None:
        """Add center line dashes to the track."""
        spine_len = len(spine)
        index = 0
        
        while index < spine_len - 10:
            if random.random() > DASH_PROBABILITY:
                p1 = spine[index]
                p2 = spine[index + 8]
                pygame.draw.line(
                    surface, YELLOW,
                    (int(p1[0]), int(p1[1])),
                    (int(p2[0]), int(p2[1])),
                    2
                )
            
            index += DASH_SPACING_BASE + random.randint(*DASH_RANDOM_GAP)

    def _add_edges(self, surface: pygame.Surface, spine: List[Tuple[float, float]]) -> None:
        """Add edge line markings on both sides of the road."""
        spine_len = len(spine)
        
        for side_sign in (+1, -1):
            offset = (ROAD_HALF_WIDTH - 6) * side_sign
            index = 0
            
            while index < spine_len - 5:
                if random.random() > EDGE_PROBABILITY:
                    nx1, ny1 = self._get_normal_at_index(index)
                    nx2, ny2 = self._get_normal_at_index(index + 5)
                    
                    x1 = int(spine[index][0] + nx1 * offset)
                    y1 = int(spine[index][1] + ny1 * offset)
                    x2 = int(spine[index + 5][0] + nx2 * offset)
                    y2 = int(spine[index + 5][1] + ny2 * offset)
                    
                    pygame.draw.line(surface, WHITE, (x1, y1), (x2, y2), 1)
                
                index += EDGE_SEGMENT_LENGTH + random.randint(*EDGE_RANDOM_GAP)

    def _add_kerbs(self, surface: pygame.Surface, spine: List[Tuple[float, float]]) -> None:
        """Add kerb colorations at sections around the track."""
        spine_len = len(spine)
        kerb_offset = ROAD_HALF_WIDTH - 6
        
        for fraction in KERB_SECTIONS:
            base_index = int(fraction * spine_len)
            
            for stripe_idx in range(KERB_LENGTH):
                spine_idx = (base_index + stripe_idx) % spine_len
                cx, cy = spine[spine_idx]
                nx, ny = self._get_normal_at_index(spine_idx)
                
                color = KRED if stripe_idx % 2 == 0 else KWHT
                
                x1 = int(cx + nx * kerb_offset)
                y1 = int(cy + ny * kerb_offset)
                x2 = int(cx + nx * (kerb_offset + 10))
                y2 = int(cy + ny * (kerb_offset + 10))
                
                pygame.draw.line(surface, color, (x1, y1), (x2, y2), 3)

    def _add_potholes(self, surface: pygame.Surface, spine: List[Tuple[float, float]]) -> None:
        """Add random potholes to the road surface and store collision rects."""
        spine_len = len(spine)
        rng = random.Random(42)  # Deterministic randomness for reproducibility
        count = 0
        attempts = 0
        
        while count < POTHOLE_TARGET_COUNT and attempts < POTHOLE_MAX_ATTEMPTS:
            attempts += 1
            spine_idx = rng.randint(0, spine_len - 1)
            cx, cy = spine[spine_idx]
            nx, ny = self._get_normal_at_index(spine_idx)
            
            # Offset perpendicular to road
            offset = rng.choice([-1, 1]) * rng.randint(*POTHOLE_DISTANCE_RANGE)
            px = int(cx + nx * offset)
            py = int(cy + ny * offset)
            
            # Check bounds
            if not (GRAIN_X_BOUNDS[0] < px < GRAIN_X_BOUNDS[1] and
                    GRAIN_Y_BOUNDS[0] < py < GRAIN_Y_BOUNDS[1]):
                continue
            
            try:
                pixel_color = surface.get_at((px, py))[:3]
                if abs(pixel_color[0] - ASPHALT[0]) > 20:
                    continue
            except IndexError:
                continue
            
            # Draw pothole
            width = rng.randint(*POTHOLE_SIZE_RANGE[0])
            height = rng.randint(*POTHOLE_SIZE_RANGE[1])
            
            pygame.draw.ellipse(surface, DARK, (px - width, py - height, width * 2, height * 2))
            pygame.draw.ellipse(surface, (80, 73, 63), 
                              (px - width, py - height, width * 2, height * 2), 1)
            
            # Store collision rectangle
            rect = pygame.Rect(px - width - 2, py - height - 2, width * 2 + 4, height * 2 + 4)
            self.pothole_rects.append(rect)
            count += 1

    def _add_bumps(self, surface: pygame.Surface, spine: List[Tuple[float, float]]) -> None:
        """Add transverse bump stripes across the track (speedbreakers)."""
        spine_len = len(spine)
        
        for fraction in BUMP_SECTIONS:
            spine_idx = int(fraction * spine_len)
            if spine_idx >= spine_len:
                continue
            
            cx, cy = spine[spine_idx]
            nx, ny = self._get_normal_at_index(spine_idx)
            
            bump_width = ROAD_HALF_WIDTH - 8
            stripe_width = bump_width * 2 / (BUMP_STRIPE_COUNT - 1)
            
            # Store collision rect for speedbreaker (full width across road)
            bump_extent = ROAD_HALF_WIDTH + 10
            rect = pygame.Rect(
                int(cx - bump_extent), int(cy - 15),
                int(bump_extent * 2), 30
            )
            self.speedbreaker_rects.append(rect)
            
            for stripe_idx in range(BUMP_STRIPE_COUNT):
                offset1 = -bump_width + stripe_idx * stripe_width
                offset2 = offset1 + stripe_width - 1
                
                color = (238, 208, 0) if stripe_idx % 2 == 0 else (34, 31, 27)
                
                x1 = int(cx + nx * offset1)
                y1 = int(cy + ny * offset1)
                x2 = int(cx + nx * offset2)
                y2 = int(cy + ny * offset2)
                
                pygame.draw.line(surface, color, (x1, y1), (x2, y2), 5)

    def _add_obstacles(self, surface: pygame.Surface, spine: List[Tuple[float, float]]) -> None:
        """Place obstacle boxes on the track."""
        spine_len = len(spine)
        self.obstacle_rects = []
        
        for fraction, side_offset, width, height, color in OBSTACLE_CONFIG:
            spine_idx = int(fraction * spine_len)
            if spine_idx >= spine_len:
                continue
            
            cx, cy = spine[spine_idx]
            nx, ny = self._get_normal_at_index(spine_idx)
            
            ox = int(cx + nx * side_offset - width // 2)
            oy = int(cy + ny * side_offset - height // 2)
            
            rect = pygame.Rect(ox, oy, width, height)
            self.obstacle_rects.append(rect)
            
            pygame.draw.rect(surface, color, rect, border_radius=3)
            pygame.draw.rect(surface, (14, 14, 14), rect, 1, border_radius=3)

    def _draw_startline(self, surface: pygame.Surface, spine: List[Tuple[float, float]]) -> None:
        """Draw the start/finish line on the track."""
        # Find spine point closest to spawn position
        closest_idx = min(
            range(len(spine)),
            key=lambda i: math.hypot(spine[i][0] - SPAWN_X, spine[i][1] - SPAWN_Y)
        )
        
        cx, cy = spine[closest_idx]
        nx, ny = self._get_normal_at_index(closest_idx)
        
        # Draw checkered line
        half_width = STARTLINE_TILE_COUNT * STARTLINE_TILE_SIZE / 2
        
        for tile_idx in range(STARTLINE_TILE_COUNT):
            offset1 = -half_width + tile_idx * STARTLINE_TILE_SIZE
            offset2 = offset1 + STARTLINE_TILE_SIZE
            
            color = (240, 240, 240) if tile_idx % 2 == 0 else (20, 20, 20)
            
            x1 = int(cx + nx * offset1)
            y1 = int(cy + ny * offset1)
            x2 = int(cx + nx * offset2)
            y2 = int(cy + ny * offset2)
            
            pygame.draw.line(surface, color, (x1, y1), (x2, y2), 8)

    def draw(self, screen: pygame.Surface) -> None:
        """Render the track onto the screen."""
        screen.blit(self.surface, (0, 0))