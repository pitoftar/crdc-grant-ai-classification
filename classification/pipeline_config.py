from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
OUT_DIR = BASE_DIR / 'out'

# modele a utiliser (voir models.txt)

MODEL = 'MoritzLaurer/ModernBERT-large-zeroshot-v2.0'

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
DATASET = MINI # MINI, SAMPLE or FULL

"""/!\ ↓↓↓ CHANGER LE PIPELINE ICI ↓↓↓ /!\ """
SEQ = 'tc' # tc pour titre et comité, t pour titre seulement
FINE_TUNING = 'ltd' # raw sans fine-tuning (mode par default), ltd pour limitee, finet avec fine-tuning, ltd+finet pour limitee avec fine-tuning
SCOPE = scope_map[DATASET]