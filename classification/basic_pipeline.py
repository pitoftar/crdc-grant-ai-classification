# --- preambule ---

import os
import re
import logging
from transformers import pipeline
from datetime import datetime
import pandas as pd
import numpy as np
import csv

logger = logging.getLogger(__name__)
logging.basicConfig(filename='classification_pipeline.log', level=logging.INFO)

# pipeline pour classification

MODEL = 'MoritzLaurer/deberta-v3-large-zeroshot-v2.0'
classifier = pipeline("zero-shot-classification", model=MODEL)

# --- donnees projets ---

crdc = pd.read_csv(
    '../data/crdc-full-encoder.csv',
    names=['single_encoder',
        'code_d',
        'code_g',
        'code_c', 
        'code_sc',
        'division', 
        'group', 
        'class', 
        'subclass']
    )

# si le csv comporte des titres, enlever la premiere ligne du df
# pour eviter de contaminer les donnees

crdc.drop(
    index=crdc.index[0],
    axis=0,
    inplace=True
)

# stocker les divisions et groupes uniques dans un dictionnaire

def codes_uniques(
    dataframe: pd.DataFrame,
    index,
    colonne) -> dict:
    """Renvoie un dictionnaire a partir d'un dataframe sous
    la forme {index: colonne} en purgeant les entrees vides.
    """
    dataframe = dataframe.dropna()
    return dataframe.set_index(index)[colonne].to_dict()

# fonction simple pour obtenir une liste de valeurs uniques
# d'apres des colonnes dans un dictionnaire de dataframes
# Source - https://stackoverflow.com/a/66912198
# Posted by Daniel Warfield et modifie par ASA
# Retrieved 2026-08-08, License - CC BY-SA 4.0

def liste_colonne(
    dictionnaire: dict[pd.DataFrame],
    colonne: str) -> dict:

    """Retourne, sous la forme d'un dictionnaire,
    les valeurs contenues dans la colonne specifiee
    par la variable "colonne" au travers d'un dictionnaire
    de dataframes specifie par la variable "dictionnaire".
    """

    for clef, valeur in dictionnaire.items():
        dictionnaire[clef] = valeur[colonne].unique().tolist()
    
    return dictionnaire

divisions = codes_uniques(crdc, 'code_d', 'division')
divisions_inverse = codes_uniques(crdc, 'division', 'code_d')
groupes = codes_uniques(crdc, 'code_g', 'group')
groupes_inverse = codes_uniques(crdc, 'group', 'code_g')
classes = codes_uniques(crdc, 'code_c', 'class')
sous_classes = codes_uniques(crdc, 'code_sc', 'subclass')

crdc_div = {code: discipline for code, discipline in crdc.dropna().groupby('division')}
crdc_gr = {code: discipline for code, discipline in crdc.dropna().groupby('group')}
crdc_cls = {code: discipline for code, discipline in crdc.dropna().groupby('class')}

# fonction pour rendre les resultats plus manipulables (actuellement pas utilisee)

def offload_dans_dict(
    liste_dicts_de_resultats: list,
    NIVEAU: str) -> list:

    """A partir d'une liste de dictionnaires obtenus comme
    resultats de la fonction classifier(), retourne une liste
    de dictionnaires ou les paires clef/valeurs sont des tuples (str, str).

    La variable "liste_dicts_de_resultats" fournie a la fonction
    correspond a une liste de dictionnaire structuree comme suit:
        [{sequence: str, labels: [], scores: []},
        {sequence: str, labels: [], scores: []}...].

    La variable "NIVEAU" est un string qui traduit le niveau
    de classification afin d'identifier les variables adequatement
    (e.g. 'div', 'gr', etc.). Il alimente la nomenclature des clefs
    (p.ex si NIVEAU='gr', les variables seront associees aux clefs
    labels_gr_1, labels_gr_2 ... labels_gr_n).
    """

    liste = []

    for resultat in liste_dicts_de_resultats:
        rangee = {}
        for k, v in resultat.items():
            if isinstance(v, list):
                for idx, val in enumerate(v, start=1):
                    rangee.update({f"{k}_{NIVEAU}_{idx}": val})
            else:
                rangee.update({k : v})
        liste.append(rangee)

    return liste

# fonction de classification simple (non-limitee par le niveau superieur)

def classificateur_simple(
    sequences: list,
    categories,
    multi_label_bool: bool=False) -> list:

    """Retourne une liste de dictionnaires de resultats tires d'une
    variante simple de la fonction classifier() de transformers.

    La variable "sequences" fournie a la fonction est une liste de
    strings a classifier.

    La variable "categories" est une liste des classes a associer aux
    sequences OU un dictionnaire ou les valeurs representent les
    classes a associer aux sequences.

    La variable "multi_label_bool" est un booleen (valeur par defaut = False)
    qui indique si les probabilites doivent etre softmaxees
    individuellement entre les categories (plusieurs categories
    possibles, True) ou qu'elles doivent equivaloir a un total de
    1 (une seule categorie possible).
    """

    liste = []

    if isinstance(categories, list):
        cat = categories
    elif isinstance(categories, dict):
        cat = list(categories.values())
    else:
        raise Exception("La variable 'categories' doit etre une liste ou un dictionnaire.")
        # ou gerer avec logger?

    for i, seq in enumerate(sequences):

        resultat = classifier(
            seq,
            cat,
            multi_label=multi_label_bool
        )

        liste.append(resultat)

        logger.info(f"Grant #{i+1} classification DONE\t\t\t{seq}")
        # ajouter logging.basicConfig(filename='classification_pipeline.log', level=logging.INFO)
        # a def main() pour creer document de log
    
    return liste

def classificateur_complexe(
    resultats: list,
    categories: dict,
    multi_label_bool: bool = True) -> list:

    """Retourne une liste de resultats classifies selon une division
    inferieure en limitant la classification aux sous-groupes
    de la premiere etiquette.
    Par exemple, au moment de classer un projet categorise selon la
    division 'Social sciences' (RDF50), le script ne considerera que
    les groupes appartenant a cette division (RDF50X).

    La variable "resultats" fournie a la fonction
    correspond a une liste de dictionnaire structuree comme suit:
        [{sequence: str, labels: [], scores: []},
        {sequence: str, labels: [], scores: []}...].

    La variable "categories" est un dictionnaire ou chaque clef est
    la representation textuelle de la categorie superieure et les valeurs
    sont une liste des categories inferieures, p. ex. :
        'Social sciences': ['Psychology and cognitive sciences', 'Economics and
        business administration', 'Education', 'Sociology and related studies', ...],
    potentiellement issue de la fonction liste_colonne().

    La variable "multi_label_bool" est un booleen (valeur par defaut = True)
    qui indique si les probabilites doivent etre softmaxees
    individuellement entre les categories (plusieurs categories
    possibles, True) ou qu'elles doivent equivaloir a un total de
    1 (une seule categorie possible).
    """

    liste = []

    for i, resultat in enumerate(resultats):
        categorie_probable = resultat['labels'][0] # bug ici : string indices must be integers
        score_cat_no_1 = resultat['scores'][0]
        titre = resultat['sequence']

        resultat_limite = classifier(titre, categories[categorie_probable], multi_label=multi_label_bool)

        liste.append(resultat_limite)
        logger.info(f"Grant #{i+1} limited group-level DONE\t\t\t{titre}")

    return liste

def structurer_resultats(
    resultats_classification: list[dict],
    dict_idu: dict,
    limite: int,
    NIVEAU: str = None) -> pd.DataFrame:

    """Retourne un dataframe avec les colonnes organisees
    selon l'identifiant unique, la categorie et le score.

    La variable 'resultats_classification' est une liste de dicts
    resultant de la fonction classifier() de HuggingFace, ou chaque
    dict est compose des clefs {sequence: str, labels: [], scores: []}.

    La variable 'dict_idu' represente les categories du CCRD
    sous forme de dictionnaire "inverse", c'est-a-dire que le
    code unique et la description textuelle sont organisees sous
    le format {description: clef}.

    La variable 'limite' est un chiffre pour limiter le nombre
    de categories inscrites dans le dataframe.

    La variable "NIVEAU" est un string qui traduit le niveau
    de classification afin d'identifier les variables adequatement
    (e.g. 'div', 'gr', etc.). Il alimente la nomenclature des clefs
    (p.ex si NIVEAU='gr', les variables seront associees aux clefs
    label_gr_1, label_gr_2 ... label_gr_n). Elle est facultative.
    """

    rangees = []

    prefixe = f"{NIVEAU}_" if NIVEAU else ''

    for resultat in resultats_classification:
        rangee = {"sequence": resultat["sequence"]}

        for idx, (categorie, score) in enumerate(
            zip(
                resultat["labels"],
                resultat["scores"]),
            start=1
        ):
            if idx > limite:
                break
            rangee[f"code_{prefixe}{idx}"] = dict_idu[categorie]
            rangee[f"label_{prefixe}{idx}"] = categorie
            rangee[f"score_{prefixe}{idx}"] = score

        rangees.append(rangee)
    
    return pd.DataFrame(rangees)

# --- fonctions de classification de plus haut niveau ---

def classification_large(): # potentiellement customiser davantage
    # classification division

    premier_niveau = classificateur_simple(
        sequences=titres_comites,
        categories=divisions,
        multi_label_bool=False
    )

    # classification groupe

    deuxieme_niveau = classificateur_simple(
        sequences=premier_niveau,
        categories=groupes,
        multi_label_bool=True
    )

    # top 3 div top 5 gr

    top_n_premier_niveau = structurer_resultats(
        resultats_classification=premier_niveau,
        dict_idu=divisions_inverse,
        limite=3,
        NIVEAU='div'
    )

    top_n_deuxieme_niveau = structurer_resultats(
        resultats_classification=deuxieme_niveau,
        dict_idu=groupes_inverse,
        limite=5,
        NIVEAU='div'
    )

    resultat_final = top_n_deuxieme_niveau.merge(top_n_deuxieme_niveau, on='sequence')

    return resultat_final

def classification_limitee():
    # classification division

    premier_niveau = classificateur_simple(
        sequences=titres_comites,
        categories=divisions,
        multi_label_bool=False
    )

    # classification groupe

    deuxieme_niveau = classificateur_complexe(
        resultats=premier_niveau,
        categories=groupes_par_div,
        multi_label_bool=True
    )

    # top 1 div top 3 gr

    top_n_premier_niveau = structurer_resultats(
        resultats_classification=premier_niveau,
        dict_idu=divisions_inverse,
        limite=1,
        NIVEAU='div'
    )

    top_n_deuxieme_niveau = structurer_resultats(
        resultats_classification=deuxieme_niveau,
        dict_idu=groupes_inverse,
        limite=3,
        NIVEAU='gr'
    )

    resultat_final = top_n_premier_niveau.merge(top_n_deuxieme_niveau, on='sequence')

    return resultat_final


# dictionnaire sous la forme en plein texte {division: [groupe 1, groupe 2 ... groupe n]}
groupes_par_div = liste_colonne(crdc_div, 'group')
cls_par_gr = liste_colonne(crdc_gr, 'class')
subcls_par_cls = liste_colonne(crdc_cls, 'subclass')

# initialiser liste des titres seuls et initialiser liste des
# titres avec comites entre parenthese (le cas echeant)

MINI = '../data/smaller_sample.csv'
SAMPLE = '../data/sample.csv'
FULL = '../data/projets_comites_complets-ENFR.csv'

"""/!\ ↓↓↓ CHANGER LA SOURCE DES DONNEES ICI ↓↓↓ /!\ """
DATASET = MINI

scope_map = {MINI: 'mini', SAMPLE: 'sample', FULL: 'full'}

dtfrm = pd.read_csv(DATASET, sep=';', names=['comite_en', 'comite_fr', 'titre'])

projets = list(dtfrm.T.to_dict().values())
k = 'titre'
c = 'comite_en'

titres = [projet.get(k) for projet in projets if k in projet]

titres_comites = []

for projet in projets:
    if str(projet[c]) == 'nan':
        titres_comites.append(projet[k])
    else:
        titres_comites.append(f'{projet[k]} ({projet[c]})')

# --- classification des projets selon la division ---

# passage dans le classificateur

merged_datfra = classification_limitee() # runner un debug la-dessus

# resultats_division = classificateur_simple(
#     sequences=titres_comites,
#     categories=divisions,
#     multi_label_bool=False
# )

# # classification groupes limitee

# resultats_groupe_limite_par_div = classificateur_complexe(
#     resultats=resultats_division,
#     categories=groupes_par_div,
#     multi_label_bool=True
# )

# # classification groupes totale

# resultats_groupes = classificateur_simple(
#     sequences=titres_comites,
#     categories=groupes,
#     multi_label_bool=True
# )

# # --- nettoyage des resultats ---

# div_top_3 = structurer_resultats(
#     resultats_classification=resultats_division,
#     dict_idu=divisions_inverse,
#     limite=3,
#     NIVEAU='div'
# )

# groupes_top_5 = structurer_resultats(
#     resultats_classification=resultats_groupe_limite_par_div,
#     dict_idu=groupes_inverse,
#     limite=5,
#     NIVEAU='gr'
# )

# merged_datfra = div_top_3.merge(groupes_top_5, on='sequence')

# raise SystemExit

# ajuster le titre du document de sortie en fonction du traitement de la classification

now = datetime.now().strftime('%Y%m%d-%H%M')

if not os.path.exists(f"../out/{re.sub('/', '-', MODEL)}/"):
    os.makedirs(f"../out/{re.sub('/', '-', MODEL)}/")

SEQ = 'tc' # tc pour titre et comité, t pour titre seulement
FINE_TUNING = 'ltd' # raw sans fine-tuning, ltd pour limitee, finet avec fine-tuning
LEVEL = 'gr' # div pour division, gr pour groupe, cls pour classe
SCOPE = scope_map[DATASET]

merged_datfra.to_csv(
    f"../out/{re.sub('/', '-', MODEL)}/{now}_{SEQ}_{FINE_TUNING}_{LEVEL}_{SCOPE}.csv",
    sep=';',
    mode='w',
    quotechar='"'
)