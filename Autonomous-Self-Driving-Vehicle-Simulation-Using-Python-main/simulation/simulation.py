"""
Runtime and visualization code for the simulation.

This file manages the Pygame loop, generation updates, and the HUD that
shows the car statistics and fitness graph.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path when this file is executed directly.
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pygame
import numpy as np
import matplotlib.pyplot as plt
from config import (
    COL_BEST_CAR,
    COL_ROAD_EDGE,
    COL_SENSOR,
    COL_TEXT,
    COL_TEXT_DIM,
    COL_UI_BG,
    COL_UI_ACCENT,
    FPS,
    MAX_TICKS_PER_GEN,
    POPULATION_SIZE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from car.car import Car
from car.sensors import Sensors
from ai.neural_network import NeuralNetwork
from environment.track import Track
from genetic.population import Genome, create_population_from_saved, evolve, load_best_genome
from simulation.fitness import calculate_fitness


def _draw_panel(surface: pygame.Surface, rect: pygame.Rect, fill: tuple, outline: tuple, corner: int = 10) -> None:
    pygame.draw.rect(surface, fill, rect, border_radius=corner)
    pygame.draw.rect(surface, outline, rect, 1, border_radius=corner)


def _render_text(surface: pygame.Surface, font: pygame.font.Font, text: str, position: tuple[int, int], color: tuple = COL_TEXT, align: str = "left") -> None:
    img = font.render(text, True, color)
    x, y = position
    if align == "center":
        x -= img.get_width() // 2
    elif align == "right":
        x -= img.get_width()
    surface.blit(img, (x, y))


class Button:
    def __init__(self, rect: pygame.Rect, label: str, fill: tuple, hover: tuple) -> None:
        self.rect = rect
        self.label = label
        self.fill = fill
        self.hover_fill = hover
        self.is_hover = False

    def update(self, mouse_pos: tuple[int, int]) -> None:
        self.is_hover = self.rect.collidepoint(mouse_pos)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        _draw_panel(surface, self.rect, self.hover_fill if self.is_hover else self.fill, (43, 48, 76), 8)
        _render_text(surface, font, self.label, self.rect.center, COL_TEXT, "center")

    def clicked(self, event: pygame.event.Event) -> bool:
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos)


class Simulation:
    HUD_WIDTH = 250
    GRAPH_HEIGHT = 118

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Autonomous Driving Evolution")
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.SysFont("Segoe UI", 22, bold=True)
        self.font_regular = pygame.font.SysFont("Segoe UI", 15)
        self.font_small = pygame.font.SysFont("Segoe UI", 13)

        self.track = Track()
        self.network = NeuralNetwork()
        self.sensor_system = Sensors()

        self.generation = 1
        self.paused = False
        self.speed_factor = 1
        self.tick_count = 0
        self.status = ""
        self.status_life = 0

        self.history_best: list[float] = []
        self.history_avg: list[float] = []
        self.history_min: list[float] = []
        self.best_ever = 0.0

        self.population = [Genome() for _ in range(POPULATION_SIZE)]
        self._reset_generation()

        px = SCREEN_WIDTH - self.HUD_WIDTH + 16
        self.button_pause = Button(pygame.Rect(px, 16, 100, 34), "⏸ Pause", (55, 60, 95), (85, 92, 125))
        self.button_speed = Button(pygame.Rect(px + 112, 16, 100, 34), "▶▶ 1×", (30, 78, 130), (62, 116, 175))
        self.button_load = Button(pygame.Rect(px, 60, 100, 34), "📂 Load", (34, 102, 62), (58, 144, 96))
        self.button_save = Button(pygame.Rect(px + 112, 60, 100, 34), "💾 Save", (102, 78, 26), (138, 108, 42))
        self.button_quit = Button(pygame.Rect(px, 104, 212, 34), "⏹ Quit", (115, 32, 32), (155, 48, 48))

    def _reset_generation(self) -> None:
        self.cars = [Car(220, 500, genome) for genome in self.population]
        for vehicle in self.cars:
            vehicle.angle = 90.0
        self.tick_count = 0

    def run(self) -> None:
        while True:
            mouse_position = pygame.mouse.get_pos()
            for button in (self.button_pause, self.button_speed, self.button_load, self.button_save, self.button_quit):
                button.update(mouse_position)

            self._process_events()
            if not self.paused:
                for _ in range(self.speed_factor):
                    self._advance_frame()

            self._render()
            self.clock.tick(FPS)

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._shutdown()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._toggle_pause()
                elif event.key == pygame.K_s:
                    self._save_current_champion()

            if self.button_pause.clicked(event):
                self._toggle_pause()
            elif self.button_speed.clicked(event):
                self._rotate_speed()
            elif self.button_load.clicked(event):
                self._load_genome()
            elif self.button_save.clicked(event):
                self._save_current_champion()
            elif self.button_quit.clicked(event):
                self._shutdown()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.button_pause.label = "▶ Resume" if self.paused else "⏸ Pause"

    def _rotate_speed(self) -> None:
        self.speed_factor = {1: 2, 2: 4, 4: 1}[self.speed_factor]
        self.button_speed.label = f"▶▶ {self.speed_factor}×"

    def _advance_frame(self) -> None:
        self.tick_count += 1
        active_count = 0

        for vehicle in self.cars:
            if not vehicle.alive:
                continue

            vehicle.sensors = self.sensor_system.get_readings(vehicle.pos, vehicle.angle, self.track.mask)
            controls = self.network.infer(np.array(vehicle.sensors, dtype=np.float32), vehicle.genome.weights)
            vehicle.apply_controls(controls)

            if self._vehicle_off_track(vehicle):
                vehicle.alive = False
                vehicle.genome.fitness = calculate_fitness(vehicle)
            else:
                active_count += 1

        if active_count == 0 or self.tick_count >= MAX_TICKS_PER_GEN:
            for vehicle in self.cars:
                if vehicle.alive:
                    vehicle.alive = False
                    vehicle.genome.fitness = calculate_fitness(vehicle)
            self._complete_generation()

    def _vehicle_off_track(self, vehicle: Car) -> bool:
        x = int(vehicle.pos.x)
        y = int(vehicle.pos.y)
        if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
            return True
        return bool(self.track.mask.get_at((x, y)))

    def _complete_generation(self) -> None:
        fitness_values = [genome.fitness for genome in self.population]
        if fitness_values:
            best_value = max(fitness_values)
            self.history_best.append(best_value)
            self.history_avg.append(sum(fitness_values) / len(fitness_values))
            self.history_min.append(min(fitness_values))
            self.best_ever = max(self.best_ever, best_value)

        self.population = evolve(self.population)
        self.generation += 1
        self._reset_generation()

    def _leader(self) -> Car | None:
        candidates = [vehicle for vehicle in self.cars if vehicle.alive]
        return max(candidates, key=lambda v: v.distance) if candidates else None

    def _render(self) -> None:
        self.track.draw(self.screen)
        leader = self._leader()
        for vehicle in self.cars:
            vehicle.draw(self.screen, highlight=(vehicle is leader))

        self._draw_sidebar()
        if self.status_life > 0:
            self._draw_status_message()
            self.status_life -= 1

        pygame.display.flip()

    def _draw_sidebar(self) -> None:
        left = SCREEN_WIDTH - self.HUD_WIDTH
        sidebar = pygame.Surface((self.HUD_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        sidebar.fill((*COL_UI_BG, 220))
        self.screen.blit(sidebar, (left, 0))
        pygame.draw.line(self.screen, COL_ROAD_EDGE, (left, 0), (left, SCREEN_HEIGHT), 2)

        for button in (self.button_pause, self.button_speed, self.button_load, self.button_save, self.button_quit):
            button.draw(self.screen, self.font_small)

        top = 150
        leader = self._leader()
        labels = [
            ("Generation", str(self.generation)),
            ("Active", f"{sum(1 for vehicle in self.cars if vehicle.alive)} / {POPULATION_SIZE}"),
            ("Frame", f"{self.tick_count} / {MAX_TICKS_PER_GEN}"),
            ("Best dist", f"{max((vehicle.distance for vehicle in self.cars), default=0):.0f}"),
            ("Top speed", f"{leader.speed if leader else 0:.1f}"),
            ("Lifetime", f"{self.best_ever:.1f}"),
        ]
        for index, (label, value) in enumerate(labels):
            y = top + index * 28
            _render_text(self.screen, self.font_small, label, (left + 16, y), COL_TEXT_DIM)
            _render_text(self.screen, self.font_regular, value, (SCREEN_WIDTH - 16, y), COL_TEXT, "right")

        graph_top = top + 180
        _draw_panel(self.screen, pygame.Rect(left + 16, graph_top, self.HUD_WIDTH - 32, self.GRAPH_HEIGHT + 28), (25, 28, 40), (46, 52, 79), 10)
        _render_text(self.screen, self.font_small, "Fitness trace", (left + 20, graph_top + 8), COL_TEXT_DIM)

        if len(self.history_best) >= 2:
            window_best = self.history_best[-40:]
            window_avg = self.history_avg[-40:]
            peak = max(window_best) or 1
            graph_width = self.HUD_WIDTH - 44
            start_x = left + 20
            base_y = graph_top + self.GRAPH_HEIGHT + 24

            pygame.draw.line(self.screen, COL_TEXT_DIM, (start_x, base_y), (start_x + graph_width, base_y), 1)
            pygame.draw.line(self.screen, COL_TEXT_DIM, (start_x, graph_top + 22), (start_x + graph_width, graph_top + 22), 1)

            points_best = [
                (start_x + int(i * graph_width / (len(window_best) - 1)), base_y - int(value / peak * self.GRAPH_HEIGHT))
                for i, value in enumerate(window_best)
            ]
            points_avg = [
                (start_x + int(i * graph_width / (len(window_avg) - 1)), base_y - int(value / peak * self.GRAPH_HEIGHT))
                for i, value in enumerate(window_avg)
            ]
            pygame.draw.lines(self.screen, COL_BEST_CAR, False, points_best, 2)
            pygame.draw.lines(self.screen, COL_UI_ACCENT, False, points_avg, 1)
            _render_text(self.screen, self.font_small, "Best", (left + 20, base_y + 8), COL_BEST_CAR)
            _render_text(self.screen, self.font_small, "Avg", (left + 78, base_y + 8), COL_UI_ACCENT)

        _render_text(self.screen, self.font_small, "[Space] Pause/Resume", (left + 16, SCREEN_HEIGHT - 52), COL_TEXT)
        _render_text(self.screen, self.font_small, "[S] Save genome", (left + 16, SCREEN_HEIGHT - 32), COL_TEXT)

    def _draw_status_message(self) -> None:
        rendered = self.font_regular.render(self.status, True, (20, 20, 20))
        width, height = rendered.get_size()
        box = pygame.Rect((SCREEN_WIDTH - width) // 2 - 18, SCREEN_HEIGHT - 58, width + 36, height + 20)
        _draw_panel(self.screen, box, (236, 244, 252), (102, 122, 148), 12)
        self.screen.blit(rendered, (box.left + 18, box.top + 10))

    def _save_current_champion(self) -> None:
        self.population.sort(key=lambda genome: genome.fitness, reverse=True)
        from genetic.population import save_genome
        save_genome(self.population[0])
        self._set_status("Champion genome saved.")

    def _load_genome(self) -> None:
        saved = load_best_genome()
        if not saved:
            self._set_status("No saved genome found.")
            return
        self.population = create_population_from_saved(saved)
        self.generation = 1
        self._reset_generation()
        self._set_status("Saved genome loaded.")

    def _shutdown(self) -> None:
        self._save_current_champion()
        pygame.quit()
        self._display_statistics()
        sys.exit(0)

    def _set_status(self, message: str, duration: int = 180) -> None:
        self.status = message
        self.status_life = duration

    def _display_statistics(self) -> None:
        if len(self.history_best) < 2:
            return

        generations = list(range(1, len(self.history_best) + 1))
        best = np.array(self.history_best)
        average = np.array(self.history_avg)
        minimum = np.array(self.history_min)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(generations, best, label="Best", color="#ffcc00", linewidth=2)
        ax.plot(generations, average, label="Average", color="#74b9ff", linewidth=1.5)
        ax.plot(generations, minimum, label="Min", color="#ff7675", linewidth=1)
        ax.set_title("Evolution progress")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            plt.savefig("fitness_history.png", dpi=120)
            print("Saved fitness_history.png")


def main() -> None:
    Simulation().run()


if __name__ == "__main__":
    main()
