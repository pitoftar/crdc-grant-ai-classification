from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
OUT_DIR = BASE_DIR / 'out'

# modele a utiliser (voir models.txt)

MODEL = 'MoritzLaurer/bge-m3-zeroshot-v2.0'

# scope de donnees

MINI = DATA_DIR / 'smaller_sample.csv'
SAMPLE = DATA_DIR / 'sample.csv'
FULL = DATA_DIR / 'projets_comites_complets-ENFR.csv'

scope_map = {
    MINI: 'mini',
    SAMPLE: 'sample',
    FULL: 'full'
}

"""/!\ ↓↓↓ CHANGER LA SOURCE DES DONNEES ICI ↓↓↓ /!\ """
DATASET = FULL

"""/!\ ↓↓↓ CHANGER LE PIPELINE ICI ↓↓↓ /!\ """
SEQ = 'tc' # tc pour titre et comité, t pour titre seulement
FINE_TUNING = 'raw' # raw sans fine-tuning, ltd pour limitee, finet avec fine-tuning
SCOPE = scope_map[DATASET]