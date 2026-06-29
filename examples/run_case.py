from pathlib import Path

from cl_igdt.workflow import run_case

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    run_case(project_root)
