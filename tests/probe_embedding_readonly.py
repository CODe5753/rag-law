"""Legacy probe entrypoint now uses only Gateway; no direct model reference calls."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from gateway_all_models_probe import run
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))
