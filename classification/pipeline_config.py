from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
OUT_DIR = BASE_DIR / 'out'
MODEL = 'MoritzLaurer/bge-m3-zeroshot-v2.0'