from pathlib import Path
from runpy import run_path

pkg_dir = Path(__file__).resolve().parent


def run_vis_r_emcee():
    script_pth = pkg_dir / "vis_r_emcee.py"
    run_path(str(script_pth), run_name="__main__")
