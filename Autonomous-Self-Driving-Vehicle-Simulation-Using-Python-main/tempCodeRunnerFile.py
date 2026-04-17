"""
main.py – Entry point.
Run from the project root:  python main.py
"""
from simulation.simulation import Simulation

if __name__ == "__main__":
    sim = Simulation()
    sim.run()