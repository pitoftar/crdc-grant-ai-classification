from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
OUT_DIR = BASE_DIR / 'out'

# modele a utiliser (voir models.txt)

MODEL = 'facebook/bart-large-mnli'

# scope de donnees (a transferer dans pipeline_config.py)

MINI = DATA_DIR / 'smaller_sample.csv'
SAMPLE = DATA_DIR / 'sample.csv'
FULL = DATA_DIR / 'projets_comites_complets-ENFR.csv'

"""/!\ ↓↓↓ CHANGER LA SOURCE DES DONNEES ICI ↓↓↓ /!\ """
DATASET = MINI

scope_map = {
    MINI: 'mini',
    SAMPLE: 'sample',
    FULL: 'full'
}