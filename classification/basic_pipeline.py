# --- preambule ---

import os
import re
from transformers import pipeline
from datetime import datetime
import pandas as pd
import numpy as np
import csv

# pipeline pour classification

MODEL = 'facebook/bart-large-mnli'
# classifier = pipeline("zero-shot-classification", model=MODEL)

# --- donnees projets ---

crdc = pd.read_csv('../data/crdc-full-encoder.csv', names=['single_encoder', 'code_d', 'code_g', 'code_c', 'code_sc', 'division' ,'group', 'class', 'subclass'])

# si le csv comporte des titres, enlever la premiere ligne du df
# pour eviter de contaminer les donnees

crdc.drop(index=crdc.index[0], axis=0, inplace=True)

# stocker les divisions et groupes uniques dans un dictionnaire

def codes_uniques(dataframe, index, colonne):
    """Renvoie un dictionnaire a partir d'un dataframe sous
    la forme {index: colonne}.
    """
    return dataframe.set_index(index)[colonne].to_dict()

# fonction simple pour obtenir une liste de valeurs uniques
# d'apres des colonnes dans un dictionnaire de dataframes
# Source - https://stackoverflow.com/a/66912198
# Posted by Daniel Warfield et modifie par ASA
# Retrieved 2026-08-08, License - CC BY-SA 4.0

def liste_colonne(dictionnaire, colonne):
    """Retourne, sous la forme d'un dictionnaire,
    les valeurs contenues dans la colonne specifiee
    par la variable "colonne" au travers d'un dictionnaire
    de dataframes specifie par la variable "dictionnaire".
    Les dictionnaires retournes ne comportent pas de valeurs
    nulles.
    """
    for clef, valeur in dictionnaire.items():
        dictionnaire[clef] = valeur[colonne].unique().tolist()
    for k in dictionnaire:
        valeur = dictionnaire[k]
        valeur = [valeur for valeur in dictionnaire if str(valeur != 'nan')]
    return dictionnaire

divisions = codes_uniques(crdc, 'code_d', 'division')
divisions_inverse = codes_uniques(crdc, 'division', 'code_d') # il y a surement un moyen de renverser un dictionnaire k:v en v:k
groupes = codes_uniques(crdc, 'code_g', 'group')
groupes_inverse = codes_uniques(crdc, 'group', 'code_g')
classes = codes_uniques(crdc, 'code_c', 'class')
sous_classes = codes_uniques(crdc, 'code_sc', 'subclass')

crdc_div = {code: discipline for code, discipline in crdc.groupby('division')}

groupes_par_div = liste_colonne(crdc_div, 'group')

print(groupes_par_div) # fonctionne, il faut regler les nan
print(f"{groupes_par_div['Agricultural and veterinary sciences']}\nType: {type(groupes_par_div['Agricultural and veterinary sciences'])}")

raise SystemExit

# initialiser liste des titres seuls et initialiser liste des
# titres avec comites entre parenthese (le cas echeant)

MINI = '../data/smaller_sample.csv'
SAMPLE = '../data/sample.csv'
FULL = '../data/projets_comites_complets-ENFR.csv'

DATASET = MINI # changer la source des données ici

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

resultats_division = []

# passage dans le classificateur

# definir des fonctions de classification ici

for i, titre in enumerate(titres_comites):
    resultat = classifier(titre, list(divisions.values()), multi_label=False) # multilabel false pour la classification au niveau de la division
    resultats_division.append(resultat)
    print(f"Grant #{i+1} DONE")

# deplier les listes dans le dictionnaire

output_div_net = []

for resultat in resultats_division:
    rangee = {}
    for k, v in resultat.items():
        if isinstance(v, list):
            for idx, val in enumerate(v, start=1):
                rangee.update({f"{k}_d_{idx}": val})
        else:
            rangee.update({k : v})
    output_div_net.append(rangee)

# classification groupe (a froid)

for resultat in resultats_division:
    labels = resultat['labels']
    division_probable = labels[0]
    scores = resultat['scores']
    numero_1 = scores[0]
    print(f"Pour {resultat['sequence']}, la division la plus probable est {division_probable} (certitude de {round(numero_1 * 100, 2)}%).")

raise SystemExit

resultats_groupes = []

for i, titre in enumerate(titres_comites):
    resultat_groupe = classifier(titre, list(groupes.values()), multi_label=True)
    resultats_groupes.append(resultat_groupe)


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

classification_division.to_csv(f"../out/{re.sub('/', '-', MODEL)}/{now}_{SEQ}_{FINE_TUNING}_{LEVEL}_{SCOPE}.csv", sep=';', mode='w', quotechar='"')