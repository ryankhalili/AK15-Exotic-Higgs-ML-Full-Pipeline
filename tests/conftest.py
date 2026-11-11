import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
(ROOT / '.work' / 'temp').mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(ROOT / '.work' / 'temp')
