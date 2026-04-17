# config.py - shared constants and settings
#
# This file holds the main configuration values for the simulation.
# It keeps display, physics, network, and genetic parameters together.

# Display settings
SCREEN_WIDTH: int = 1200
SCREEN_HEIGHT: int = 800
FPS: int = 60

# Car and sensor settings
SENSOR_COUNT: int = 7  # must stay odd so one ray points dead ahead
SENSOR_LENGTH: int = 150  # max ray length in pixels
CAR_RADIUS: int = 8
MAX_SPEED: float = 6.0
MIN_SPEED: float = 1.5
STEER_POWER: float = 4.5  # degrees per frame at full output

# Neural network layout
HIDDEN_NODES: int = 10

# Genome length depends on the network shape:
# input-to-hidden weights, hidden biases, hidden-to-output weights, and output biases.
GENOME_SIZE: int = (SENSOR_COUNT * HIDDEN_NODES + HIDDEN_NODES) + \
                   (HIDDEN_NODES * 2 + 2)

# Genetic algorithm settings
POPULATION_SIZE: int = 30
MUTATION_RATE: float = 0.08  # probability each weight mutates
MUTATION_STD: float = 0.15  # gaussian std for mutation noise
ELITE_COUNT: int = 4  # genomes preserved unchanged each generation
SELECTION_TOP: int = 10  # best genomes used for breeding

# Simulation runtime limits
MAX_TICKS_PER_GEN: int = 1800  # 30 s at 60 fps – prevents stuck-car stalls

# Color palette used by the renderer
COL_BG: tuple = (15, 15, 25)
COL_ROAD: tuple = (55, 60, 70)
COL_ROAD_EDGE: tuple = (70, 75, 85)
COL_INFIELD: tuple = (15, 15, 25)
COL_LANE_MARK: tuple = (200, 185, 80)
COL_OBSTACLE: tuple = (220, 60, 60)
COL_CAR_ALIVE: tuple = (50, 220, 120)
COL_CAR_DEAD: tuple = (180, 40, 40)
COL_BEST_CAR: tuple = (255, 220, 0)
COL_SENSOR: tuple = (80, 160, 220)
COL_SENSOR_HIT: tuple = (255, 100, 60)
COL_UI_BG: tuple = (32, 38, 63)
COL_UI_ACCENT: tuple = (96, 165, 255)
COL_TEXT: tuple = (235, 240, 255)
COL_TEXT_DIM: tuple = (180, 190, 210)