from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
OUT_DIR = BASE_DIR / 'out'

# modele a utiliser (voir models.txt)

MODEL = 'facebook/bart-large-mnli'

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
DATASET = MINI

"""/!\ ↓↓↓ CHANGER LE PIPELINE ICI ↓↓↓ /!\ """
SEQ = 'tc' # tc pour titre et comité, t pour titre seulement
FINE_TUNING = 'finet' # raw sans fine-tuning, finet avec fine-tuning
LEVEL = 'gr' # div pour division, gr pour groupe, cls pour classe
SCOPE = scope_map[DATASET]