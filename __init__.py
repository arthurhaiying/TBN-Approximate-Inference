from pathlib import Path
import sys

print("Init PyTAC.")
basepath = Path(__file__).resolve().parent
sys.path.append(str(basepath))