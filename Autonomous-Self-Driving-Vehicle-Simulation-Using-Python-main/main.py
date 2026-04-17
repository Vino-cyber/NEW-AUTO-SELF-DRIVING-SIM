"""
Application entry point for the autonomous driving simulation.
Launch this module from the project root directory.
"""
from simulation.simulation import Simulation


def main() -> None:
    """Create and run the simulation instance."""
    Simulation().run()


if __name__ == "__main__":
    main()
