from pathlib import Path

basepath = Path(__file__).resolve().parents[1]

cpts = basepath / Path('logs/cpts')
dots = basepath / Path('logs/dots')
exp  = basepath / Path('logs/exp')
tacs = basepath / Path('logs/tacs')
