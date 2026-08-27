# --- preambule ---

import os
import re
import logging
from transformers import pipeline
from datetime import datetime
import pandas as pd
import numpy as np
import csv

from pipeline_config import DATA_DIR, OUT_DIR, MODEL

# stocker les divisions et groupes uniques dans un dictionnaire

def inverser_dictionnaire(
    dictionnaire: dict) -> dict:

    """Renvoie un dictionnaire inverse (valeur: clef)
    par rapport a sa forme originale.
    """

    eriannoitcid = {v: k for k, v in dictionnaire.items()}

    return eriannoitcid

def codes_uniques(
    dataframe: pd.DataFrame,
    index,
    colonne) -> dict:
    """Renvoie un dictionnaire a partir d'un dataframe sous
    la forme {index: colonne} en purgeant les entrees vides.
    """
    dataframe = dataframe.dropna()
    return dataframe.set_index(index)[colonne].to_dict()

# codes uniques pour categories verboses (abandonne)

def codes_uniques_verbose(
    niveau: dict) -> dict:

    """Retourne un dictionnaire qui applique un referencement
    croise sur les categories verboses (fine-tuning maison) et
    les codes uniques du CCRD.

    La variable 'niveau' est le dictionnaire des variables qui
    indique quel niveau (division, groupe ou classe)
    est concerne.
    """

    if niveau == div_verbose or niveau == div_sscls:
        mapping = divisions_inverse
    elif niveau == gr_verbose or niveau == gr_sscls:
        mapping = groupes_inverse
    elif niveau == cls_verbose:
        mapping = classes_inverse
    else:
        raise ValueError(f"Le niveau doit etre l'un des dictionnaires de categories affinees")

    dictionnaire = {}

    for key, value in mapping.items():
        if key in niveau:
            dictionnaire[niveau[key]] = value

    return dictionnaire

def def_niveau( # [WIP]
    donnees: dict) -> str:

    """Deduit le niveau a inscrire dans le fichier de sortie
    a partir du jeu de donnees fourni en entree.
    """

    sample = next(iter(donnees))

    collection_div = [value for d in [divisions, divisions_inverse, div_verbose, div_sscls] for value in d.values()]
    collection_gr = [value for d in [groupes, groupes_par_div, groupes_inverse, gr_verbose, gr_sscls] for value in d.values()]
    collection_cls = [value for d in [classes, classes_inverse, cls_par_gr, cls_verbose] for value in d.values()]
    collection_subscls = [value for d in [sous_classes, subcls_par_cls] for value in d.values()]

    if sample in collection_div:
        nivo = 'div'
    elif sample in collection_gr:
        nivo = 'gr'
    elif sample in collection_cls:
        nivo = 'cls'
    elif sample in collection_subscls:
        nivo = 'subcls'
    else:
        raise Exception("Impossible de determiner le niveau de classification")

    return nivo

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

def titres_projets(
    source_donnees: pd.DataFrame,
    col_titre: str,
    col_comite: str = None) -> list:

    """Retourne une liste des titres de projets pour
    classification (incluant l'information sur les comites)
    a partir d'un dataframe.

    Les variables "col_titre" et "col_comite" correspondent
    respectivement a la colonne ou l'on s'attend a trouver
    le titre et le nom du comite des projets a classifier.
    """

    liste_donnees = list(source_donnees.T.to_dict().values())

    if col_comite is None:
        titres = [liste_donnees.get(col_titre) for element in liste_donnees if col_titre in element]
    else:
        titres = []
        
        for element in liste_donnees:
            if str(element[col_comite]) == 'nan':
                titres.append(element[col_titre])
            else:
                titres.append(f"{element[col_titre]} ({element[col_comite]})")

    return titres

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

# fonction pour rencherir les categories

def categories_verboses(
    df_dict: dict[pd.DataFrame],
    colonne: str) -> dict:

    """Retourne un dictionnaire de categories enrichies de
    leurs sous-categories a partir d'un dataframe sous
    la forme
        {'Categorie 1' : 'Categorie 1 (includes sous-categorie 1;
        sous- categorie 2; ... sous-categorie n)',
        'Categorie 2' : 'Categorie 2 (includes sous-categorie 1;
        sous-categorie 2; ... sous-categorie n)'}.

    La variable "df_dict" correspond a un dictionnaire
    de dataframes ou chaque clef est une categorie.

    La variable "colonne" correspond a la colonne du
    df qui contient les sous-categories.
    """

    dictionnaire = {}

    for key, df in df_dict.items():
        ar = np.array(df[colonne].values)
        ar = np.unique(ar)
        ar = '; '.join(ar)
        string = f"{key} (includes {ar})"
        dictionnaire.update({key: string})

    return dictionnaire

# fonction de classification simple (non-limitee par le niveau superieur)

def classificateur_simple(
    sequences: list,
    categories,
    multi_label_bool: bool = False) -> list:

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
        categorie_probable = resultat['labels'][0]
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

    niveau = def_niveau(
        donnees=dict_idu
    )

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
            if niveau == 'div' and categorie in div_verbose.values():
                rangee[f"label_{prefixe}{idx}"] = divisions[dict_idu[categorie]]
            elif niveau == 'gr' and categorie in gr_verbose.values():
                rangee[f"label_{prefixe}{idx}"] = groupes[dict_idu[categorie]]
            elif niveau == 'cls' and categorie in cls_verbose.values():
                rangee[f"label_{prefixe}{idx}"] = classes[dict_idu[categorie]]
            else:
                rangee[f"label_{prefixe}{idx}"] = categorie
            rangee[f"score_{prefixe}{idx}"] = score

        rangees.append(rangee)
    
    return pd.DataFrame(rangees)

# --- fonctions de classification de plus haut niveau ---

def classification_large(
    sequences: list,
    divisions: dict,
    groupes: dict) -> pd.DataFrame :

    """Retourne un dataframe comprenant les projets classifies
    a deux niveaux (p. ex. divisions et groupes) independamment
    l'un de l'autre.
    """

    # classification division

    premier_niveau = classificateur_simple(
        sequences=sequences,
        categories=divisions,
        multi_label_bool=False
    )

    # classification groupe

    deuxieme_niveau = classificateur_simple(
        sequences=sequences,
        categories=groupes,
        multi_label_bool=True
    )

    # inversion des dictionnaires de categories pour mapping

    inv_div = inverser_dictionnaire(divisions)

    inv_gr = inverser_dictionnaire(groupes)

    # top 3 div top 5 gr

    top_n_premier_niveau = structurer_resultats(
        resultats_classification=premier_niveau,
        dict_idu=inv_div,
        limite=3,
        NIVEAU='div'
    )

    top_n_deuxieme_niveau = structurer_resultats(
        resultats_classification=deuxieme_niveau,
        dict_idu=inv_gr,
        limite=5,
        NIVEAU='gr'
    )

    resultat_final = top_n_premier_niveau.merge(top_n_deuxieme_niveau, on='sequence')

    return resultat_final

def classification_limitee(
    sequences: list,
    divisions: dict,
    groupes_par_div: dict) -> pd.DataFrame :
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

    # inversion des dictionnaires de categories pour mapping

    inv_div = inverser_dictionnaire(divisions)

    inv_gr = inverser_dictionnaire(groupes_par_div)

    # top 1 div top 3 gr

    top_n_premier_niveau = structurer_resultats(
        resultats_classification=premier_niveau,
        dict_idu=inv_div,
        limite=1,
        NIVEAU='div'
    )

    top_n_deuxieme_niveau = structurer_resultats(
        resultats_classification=deuxieme_niveau,
        dict_idu=inv_gr,
        limite=3,
        NIVEAU='gr'
    )

    resultat_final = top_n_premier_niveau.merge(top_n_deuxieme_niveau, on='sequence')

    return resultat_final

def classification_affinee_large(
    sequences: list,
    categories_generales: dict,
    categories_specifiques: dict = None) -> pd.DataFrame:

    """Retourne un dataframe comportant les resultats d'une
    classification affinee (ou les categories sont enrichies
    de leurs sous-categories).
    La deuxieme classification est effectuee sans egard pour
    le premier niveau de classification.
    """

    # classification premier niveau

    premier_niveau = classificateur_simple(
        sequences=sequences,
        categories=categories_generales,
        multi_label_bool=False
    )

    cat_idu_1 = codes_uniques_verbose(
        niveau=categories_generales
    )

    top_n_premier_niveau = structurer_resultats(
        resultats_classification=premier_niveau,
        dict_idu=cat_idu_1,
        limite=3,
        NIVEAU='div' # ajuster avec la fonction def_niveau()
    )

    # classification deuxieme niveau

    if categories_specifiques is not None:

        deuxieme_niveau = classificateur_simple(
            sequences=sequences,
            categories=categories_specifiques,
            multi_label_bool=True
        )

        cat_idu_2 = codes_uniques_verbose(
            niveau=categories_specifiques
        )

        top_n_deuxieme_niveau = structurer_resultats(
            resultats_classification=deuxieme_niveau,
            dict_idu=cat_idu_2,
            limite=5,
            NIVEAU='gr' # ajuster avec foncion def_niveau()
        )

        resultat_final = top_n_premier_niveau.merge(top_n_deuxieme_niveau, on='sequence')

    else:
        resultat_final = top_n_premier_niveau

    return resultat_final

def classification_affinee_limitee(
    sequences: list,
    categories_generales: dict,
    categories_specifiques: dict) -> pd.DataFrame:

    """Retourne un dataframe comportant les resultats d'une
    classification affinee (ou les categories sont enrichies
    de leurs sous-categories).
    La deuxieme classification est effectuee selon les resultats
    du premier niveau de classification.
    """

    # classification premier niveau

    premier_niveau = classificateur_simple(
        sequences=sequences,
        categories=categories_generales,
        multi_label_bool=False
    )

    cat_idu_1 = codes_uniques_verbose(
        niveau=categories_generales
    )

    top_n_premier_niveau = structurer_resultats(
        resultats_classification=premier_niveau,
        dict_idu=cat_idu_1,
        limite=1,
        NIVEAU='div' # ajuster avec la fonction def_niveau()
    )

    # classification deuxieme niveau

    deuxieme_niveau = classificateur_complexe(
        sequences=sequences,
        categories=
    )














# --- MAIN ---

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{OUT_DIR}/classification_pipeline.log', level=logging.INFO)

# pipeline pour classification

classifier = pipeline("zero-shot-classification", model=MODEL)

# --- donnees projets ---

crdc = pd.read_csv(
    DATA_DIR / 'crdc-full-encoder.csv',
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

# dictionnaires a partir de dataframe

divisions = codes_uniques(crdc, 'code_d', 'division')
divisions_inverse = codes_uniques(crdc, 'division', 'code_d')
groupes = codes_uniques(crdc, 'code_g', 'group')
groupes_inverse = codes_uniques(crdc, 'group', 'code_g')
classes = codes_uniques(crdc, 'code_c', 'class')
classes_inverse = codes_uniques(crdc, 'class', 'code_c')
sous_classes = codes_uniques(crdc, 'code_sc', 'subclass')

# codes individuels pour les sous-groupes de chaque division

crdc_div = {code: discipline for code, discipline in crdc.dropna().groupby('division')}
crdc_gr = {code: discipline for code, discipline in crdc.dropna().groupby('group')}
crdc_cls = {code: discipline for code, discipline in crdc.dropna().groupby('class')}

div_verbose = categories_verboses(
    df_dict=crdc_div,
    colonne='group'
)

gr_verbose = categories_verboses(
    df_dict=crdc_gr,
    colonne='class'
)

div_sscls = categories_verboses(
    df_dict=crdc_div,
    colonne='subclass'
)

gr_sscls = categories_verboses(
    df_dict=crdc_gr,
    colonne='subclass'
)

cls_verbose = categories_verboses(
    df_dict=crdc_cls,
    colonne='subclass'
)

# dictionnaire sous la forme en plein texte {division: [groupe 1, groupe 2 ... groupe n]}

groupes_par_div = liste_colonne(crdc_div, 'group')
cls_par_gr = liste_colonne(crdc_gr, 'class')
subcls_par_cls = liste_colonne(crdc_cls, 'subclass')

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

# initialiser liste des titres seuls et initialiser liste des
# titres avec comites entre parenthese (le cas echeant)

dtfrm = pd.read_csv(DATASET, sep=';', names=['comite_en', 'comite_fr', 'titre'])

titres_comites = titres_projets(
    source_donnees=dtfrm,
    col_titre='titre',
    col_comite='comite_en'
)


# --- classification des projets selon la division ---

# passage dans le classificateur

# merged_datfra = classification_large(
#     sequences=titres_comites,
#     divisions=divisions,
#     groupes=groupes
# )

# merged_datfra = classification_limitee(
#     sequences=titres_comites,
#     divisions=divisions,
#     groupes_par_div=groupes_par_div
# )

# merged_datfra = classification_affinee_large(
#     sequences=titres_comites,
#     categories_generales=div_verbose,
#     categories_specifiques=gr_verbose
# )

# merged_datfra = classification_affinee_limitee(
#     sequences=titres_comites,
#     categories_generales=div_verbose,
#     categories_specifiques= # remplir ici, il me faut un groupe par div mais verbose, help
# )

# ajuster le titre du document de sortie en fonction du traitement de la classification

now = datetime.now().strftime('%Y%m%d-%H%M')

if not os.path.exists(f"{OUT_DIR}/{re.sub('/', '-', MODEL)}/"):
    os.makedirs(f"{OUT_DIR}/{re.sub('/', '-', MODEL)}/")

SEQ = 'tc' # tc pour titre et comité, t pour titre seulement
FINE_TUNING = 'finet' # raw sans fine-tuning, ltd pour limitee, finet avec fine-tuning
LEVEL = 'gr' # div pour division, gr pour groupe, cls pour classe
SCOPE = scope_map[DATASET]

merged_datfra.to_csv(
    f"{OUT_DIR}/{re.sub('/', '-', MODEL)}/{now}_{SEQ}_{FINE_TUNING}_{LEVEL}_{SCOPE}.csv",
    sep=';',
    mode='w',
    quotechar='"'
)