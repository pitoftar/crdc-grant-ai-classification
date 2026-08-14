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

MODEL = 'cross-encoder/nli-deberta-v3-base'
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
    dictionnaire: dict,
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

# fonction pour rendre les resultats plus manipulables

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

        logger.info(f"Grant #{i+1} classification DONE")
        # ajouter logging.basicConfig(filename='classification_pipeline.log', level=logging.INFO)
        # a def main() pour creer document de log
    
    return liste

def structurer_resultats(
    resultats_classification: list[dict],
    dict_idu: dict,
    limite: int,
    NIVEAU=None) -> pd.DataFrame:
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

        for rang, (categorie, score) in enumerate(zip(resultat["labels"], resultat["scores"]), start=1):
            if rang > limite:
                break
            rangee[f"code_{prefixe}{rang}"] = dict_idu[categorie]
            rangee[f"label_{prefixe}{rang}"] = categorie
            rangee[f"score_{prefixe}{rang}"] = score

        rangees.append(rangee)
    
    return pd.DataFrame(rangees)

# fonction pour classificateur externe a determiner
# def classificateur_complexe():

# dictionnaire sous la forme en plein texte {division: [groupe 1, groupe 2 ... groupe n]}
groupes_par_div = liste_colonne(crdc_div, 'group')

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

resultats_division = classificateur_simple(
    sequences=titres_comites,
    categories=divisions,
    multi_label_bool=False
)

# deplier les listes dans le dictionnaire

output_div_net = offload_dans_dict(
    liste_dicts_de_resultats=resultats_division,
    NIVEAU='div'
)

print(f'OUTPUT DIVISIONS NET:\n{output_div_net}\n')

# classification groupes limitee

resultats_groupe_limite_par_div = []

for i, resultat in enumerate(resultats_division):
    division_probable = resultat['labels'][0]
    score_div_no_1 = resultat['scores'][0]
    titre = resultat['sequence']

    resultat_gr = classifier(titre, groupes_par_div[division_probable], multi_label=True)

    groupe_probable = resultat_gr['labels'][0]
    score_gr_no_1 = resultat_gr['scores'][0]
    resultats_groupe_limite_par_div.append(resultat_gr)
    print(f"Grant #{i+1} limited group-level DONE")

# classification groupes totale

resultats_groupes = classificateur_simple(
    sequences=titres_comites,
    categories=groupes,
    multi_label_bool=True
)

print(f"RESULTATS GROUPES:\n{resultats_groupes}\n")

output_gr_net = offload_dans_dict(
    liste_dicts_de_resultats=resultats_groupes,
    NIVEAU='gr'
)

print(f"OUTPUT GROUPES NET:\n{output_gr_net}\n")

# --- nettoyage des resultats ---

classification_division = pd.DataFrame(output_div_net)

colonnes = [
    'sequence', 'labels_d_1', 'scores_d_1', 'labels_d_2', 'scores_d_2',
    'labels_d_3', 'scores_d_3', 'labels_d_4', 'scores_d_4',
    'labels_d_5', 'scores_d_5', 'labels_d_6', 'scores_d_6'
    ]

classification_division = classification_division.reindex(columns=colonnes)

col_etiquettes = ['labels_d_1', 'labels_d_2', 'labels_d_3', 'labels_d_4', 
    'labels_d_5', 'labels_d_6']

for etiquette in col_etiquettes:
    nouvelle_col = etiquette.replace("labels", "code") # probablement pas la meilleure facon de faire
    position = classification_division.columns.get_loc(etiquette)
    classification_division.insert(position, nouvelle_col, classification_division[etiquette].map(divisions_inverse))

print(classification_division)

# ajuster le titre du document de sortie en fonction du traitement de la classification

now = datetime.now().strftime('%Y%m%d-%H%M')

if not os.path.exists(f"../out/{re.sub('/', '-', MODEL)}/"):
    os.makedirs(f"../out/{re.sub('/', '-', MODEL)}/")

SEQ = 'tc' # tc pour titre et comité, t pour titre seulement
FINE_TUNING = 'raw' # raw sans fine-tuning, finet avec fine-tuning
LEVEL = 'div' # div pour division, gr pour groupe, divgr pour groupe d'apres division
SCOPE = scope_map[DATASET]

classification_division.to_csv(
    f"../out/{re.sub('/', '-', MODEL)}/{now}_{SEQ}_{FINE_TUNING}_{LEVEL}_{SCOPE}.csv",
    sep=';',
    mode='w',
    quotechar='"'
)