"""Entry point for `python -m subforge`."""

from subforge.utils import check_dependencies
check_dependencies(gui=True)

from subforge.ui.main_window import run

if __name__ == "__main__":
    run()
