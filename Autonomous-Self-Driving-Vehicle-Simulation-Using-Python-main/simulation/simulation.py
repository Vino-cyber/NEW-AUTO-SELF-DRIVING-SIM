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
    px, py = position
    if align == "center":
        px -= img.get_width() // 2
    elif align == "right":
        px -= img.get_width()
    surface.blit(img, (px, py))


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
        color_scheme = self.hover_fill if self.is_hover else self.fill
        _draw_panel(surface, self.rect, color_scheme, (43, 48, 76), 8)
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
        self.sys_status = ""
        self.sys_status_life = 0

        self.history_best: list[float] = []
        self.history_avg: list[float] = []
        self.history_min: list[float] = []
        self.overall_highest = 0.0

        self.population = []
        iter_pop = 0
        while iter_pop < POPULATION_SIZE:
            self.population.append(Genome())
            iter_pop += 1
            
        self._renew_generation()

        x_margin = SCREEN_WIDTH - self.HUD_WIDTH + 16
        self.btn_pause = Button(pygame.Rect(x_margin, 16, 100, 34), "⏸ Pause", (55, 60, 95), (85, 92, 125))
        self.btn_speed = Button(pygame.Rect(x_margin + 112, 16, 100, 34), "▶▶ 1×", (30, 78, 130), (62, 116, 175))
        self.btn_load = Button(pygame.Rect(x_margin, 60, 100, 34), "📂 Load", (34, 102, 62), (58, 144, 96))
        self.btn_save = Button(pygame.Rect(x_margin + 112, 60, 100, 34), "💾 Save", (102, 78, 26), (138, 108, 42))
        self.btn_quit = Button(pygame.Rect(x_margin, 104, 212, 34), "⏹ Quit", (115, 32, 32), (155, 48, 48))

    def _renew_generation(self) -> None:
        self.fleet = [Car(220, 500, dna) for dna in self.population]
        for v in self.fleet:
            v.orientation = 90.0
        self.tick_count = 0

    def run(self) -> None:
        runtime_active = True
        while runtime_active:
            m_pos = pygame.mouse.get_pos()
            for btn in (self.btn_pause, self.btn_speed, self.btn_load, self.btn_save, self.btn_quit):
                btn.update(m_pos)

            self._process_events()
            
            if not self.paused:
                cycles = self.speed_factor
                while cycles > 0:
                    self._tick_forward()
                    cycles -= 1

            self._render_scene()
            self.clock.tick(FPS)

    def _process_events(self) -> None:
        events = pygame.event.get()
        for evt in events:
            if evt.type == pygame.QUIT:
                self._terminate()

            if evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_SPACE:
                    self._switch_pause()
                elif evt.key == pygame.K_s:
                    self._persist_champion()

            if self.btn_pause.clicked(evt):
                self._switch_pause()
            elif self.btn_speed.clicked(evt):
                self._swap_speed()
            elif self.btn_load.clicked(evt):
                self._retrieve_genome()
            elif self.btn_save.clicked(evt):
                self._persist_champion()
            elif self.btn_quit.clicked(evt):
                self._terminate()

    def _switch_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.label = "▶ Resume"
        else:
            self.btn_pause.label = "⏸ Pause"

    def _swap_speed(self) -> None:
        speed_map = {1: 2, 2: 4, 4: 1}
        self.speed_factor = speed_map[self.speed_factor]
        self.btn_speed.label = f"▶▶ {self.speed_factor}×"

    def _tick_forward(self) -> None:
        self.tick_count += 1
        active_vehicles = 0

        for m_car in self.fleet:
            if not m_car.is_active:
                continue

            m_car.sensor_data = self.sensor_system.get_readings(m_car.location, m_car.orientation, self.track.mask)
            converted_sensors = np.array(m_car.sensor_data, dtype=np.float32)
            actuation = self.network.infer(converted_sensors, m_car.genome.weights)
            m_car.execute_move(actuation)

            if self._out_of_bounds_check(m_car):
                m_car.is_active = False
                m_car.genome.fitness = calculate_fitness(m_car)
            else:
                active_vehicles += 1

        if active_vehicles == 0 or self.tick_count >= MAX_TICKS_PER_GEN:
            for m_car in self.fleet:
                if m_car.is_active:
                    m_car.is_active = False
                    m_car.genome.fitness = calculate_fitness(m_car)
            self._finalize_era()

    def _out_of_bounds_check(self, subject: Car) -> bool:
        cur_x = int(subject.location.x)
        cur_y = int(subject.location.y)
        
        x_invalid = cur_x < 0 or cur_x >= SCREEN_WIDTH
        y_invalid = cur_y < 0 or cur_y >= SCREEN_HEIGHT
        
        if x_invalid or y_invalid:
            return True
            
        return bool(self.track.mask.get_at((cur_x, cur_y)))

    def _finalize_era(self) -> None:
        f_scores = [gn.fitness for gn in self.population]
        if f_scores:
            top_score = max(f_scores)
            avg_score = sum(f_scores) / len(f_scores)
            bot_score = min(f_scores)
            
            self.history_best.append(top_score)
            self.history_avg.append(avg_score)
            self.history_min.append(bot_score)
            
            if top_score > self.overall_highest:
                self.overall_highest = top_score

        self.population = evolve(self.population)
        self.generation += 1
        self._renew_generation()

    def _find_frontrunner(self) -> Car | None:
        runners = [c for c in self.fleet if c.is_active]
        if len(runners) > 0:
            return max(runners, key=lambda vc: vc.distance_traveled)
        return None

    def _render_scene(self) -> None:
        self.track.draw(self.screen)
        star = self._find_frontrunner()
        
        for v in self.fleet:
            v.draw(self.screen, is_best=(v is star))

        self._render_hud()
        if self.sys_status_life > 0:
            self._render_toast_msg()
            self.sys_status_life -= 1

        pygame.display.flip()

    def _render_hud(self) -> None:
        bound_left = SCREEN_WIDTH - self.HUD_WIDTH
        sidebar_bg = pygame.Surface((self.HUD_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        sidebar_bg.fill((*COL_UI_BG, 220))
        self.screen.blit(sidebar_bg, (bound_left, 0))
        
        pygame.draw.line(self.screen, COL_ROAD_EDGE, (bound_left, 0), (bound_left, SCREEN_HEIGHT), 2)

        for gui_btn in (self.btn_pause, self.btn_speed, self.btn_load, self.btn_save, self.btn_quit):
            gui_btn.draw(self.screen, self.font_small)

        y_offset_base = 150
        star_car = self._find_frontrunner()
        
        survivors = sum([1 for c in self.fleet if c.is_active])
        max_dist = max((c.distance_traveled for c in self.fleet), default=0)
        curr_speed = star_car.velocity if star_car else 0
        
        info_pairs = [
            ("Generation", str(self.generation)),
            ("Active", f"{survivors} / {POPULATION_SIZE}"),
            ("Frame", f"{self.tick_count} / {MAX_TICKS_PER_GEN}"),
            ("Best dist", f"{max_dist:.0f}"),
            ("Top speed", f"{curr_speed:.1f}"),
            ("Lifetime", f"{self.overall_highest:.1f}"),
        ]
        
        for r_index, (tag, val) in enumerate(info_pairs):
            anchor_y = y_offset_base + (r_index * 28)
            _render_text(self.screen, self.font_small, tag, (bound_left + 16, anchor_y), COL_TEXT_DIM)
            _render_text(self.screen, self.font_regular, val, (SCREEN_WIDTH - 16, anchor_y), COL_TEXT, "right")

        plat_y = y_offset_base + 180
        _draw_panel(self.screen, pygame.Rect(bound_left + 16, plat_y, self.HUD_WIDTH - 32, self.GRAPH_HEIGHT + 28), (25, 28, 40), (46, 52, 79), 10)
        _render_text(self.screen, self.font_small, "Fitness trace", (bound_left + 20, plat_y + 8), COL_TEXT_DIM)

        if len(self.history_best) >= 2:
            lim_best = self.history_best[-40:]
            lim_avg = self.history_avg[-40:]
            chart_peak = max(lim_best) or 1
            
            box_width = self.HUD_WIDTH - 44
            p_start_x = bound_left + 20
            p_base_y = plat_y + self.GRAPH_HEIGHT + 24

            pygame.draw.line(self.screen, COL_TEXT_DIM, (p_start_x, p_base_y), (p_start_x + box_width, p_base_y), 1)
            pygame.draw.line(self.screen, COL_TEXT_DIM, (p_start_x, plat_y + 22), (p_start_x + box_width, plat_y + 22), 1)

            pts_best = [
                (p_start_x + int(step * box_width / (len(lim_best) - 1)), p_base_y - int(fitness_val / chart_peak * self.GRAPH_HEIGHT))
                for step, fitness_val in enumerate(lim_best)
            ]
            pts_avg = [
                (p_start_x + int(step * box_width / (len(lim_avg) - 1)), p_base_y - int(fitness_val / chart_peak * self.GRAPH_HEIGHT))
                for step, fitness_val in enumerate(lim_avg)
            ]
            
            pygame.draw.lines(self.screen, COL_BEST_CAR, False, pts_best, 2)
            pygame.draw.lines(self.screen, COL_UI_ACCENT, False, pts_avg, 1)
            
            _render_text(self.screen, self.font_small, "Best", (bound_left + 20, p_base_y + 8), COL_BEST_CAR)
            _render_text(self.screen, self.font_small, "Avg", (bound_left + 78, p_base_y + 8), COL_UI_ACCENT)

        _render_text(self.screen, self.font_small, "[Space] Pause/Resume", (bound_left + 16, SCREEN_HEIGHT - 52), COL_TEXT)
        _render_text(self.screen, self.font_small, "[S] Save genome", (bound_left + 16, SCREEN_HEIGHT - 32), COL_TEXT)

    def _render_toast_msg(self) -> None:
        txt_img = self.font_regular.render(self.sys_status, True, (20, 20, 20))
        w, h = txt_img.get_size()
        
        toast_box = pygame.Rect((SCREEN_WIDTH - w) // 2 - 18, SCREEN_HEIGHT - 58, w + 36, h + 20)
        _draw_panel(self.screen, toast_box, (236, 244, 252), (102, 122, 148), 12)
        self.screen.blit(txt_img, (toast_box.left + 18, toast_box.top + 10))

    def _persist_champion(self) -> None:
        self.population.sort(key=lambda gn: gn.fitness, reverse=True)
        from genetic.population import save_genome
        save_genome(self.population[0])
        self._dispatch_toast("Champion genome saved.")

    def _retrieve_genome(self) -> None:
        db_gn = load_best_genome()
        if db_gn is None:
            self._dispatch_toast("No saved genome found.")
            return
        self.population = create_population_from_saved(db_gn)
        self.generation = 1
        self._renew_generation()
        self._dispatch_toast("Saved genome loaded.")

    def _terminate(self) -> None:
        self._persist_champion()
        pygame.quit()
        self._plot_metrics()
        sys.exit(0)

    def _dispatch_toast(self, msg: str, timeout: int = 180) -> None:
        self.sys_status = msg
        self.sys_status_life = timeout

    def _plot_metrics(self) -> None:
        if len(self.history_best) < 2:
            return

        idx_arr = list(range(1, len(self.history_best) + 1))
        pts_b = np.array(self.history_best)
        pts_a = np.array(self.history_avg)
        pts_m = np.array(self.history_min)

        plt.style.use("dark_background")
        graph_fig, main_ax = plt.subplots(figsize=(10, 5))
        
        main_ax.plot(idx_arr, pts_b, label="Best", color="#ffcc00", linewidth=2)
        main_ax.plot(idx_arr, pts_a, label="Average", color="#74b9ff", linewidth=1.5)
        main_ax.plot(idx_arr, pts_m, label="Min", color="#ff7675", linewidth=1)
        
        main_ax.set_title("Evolution progress")
        main_ax.set_xlabel("Generation")
        main_ax.set_ylabel("Fitness")
        main_ax.grid(alpha=0.25)
        main_ax.legend(loc="best")
        
        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            plt.savefig("fitness_history.png", dpi=120)
            print("Saved fitness_history.png")


def main() -> None:
    session = Simulation()
    session.run()


if __name__ == "__main__":
    main()
