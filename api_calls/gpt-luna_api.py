import json
import time
from datetime import datetime
import pandas as pd
from pathlib import Path
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # récupérer la clef dans une variable d'environnement

# variables

MODEL = "gpt-6-luna"
EFFORT = "low"
BATCH_SIZE = 25 # ?
RESPONSES_DIR = Path("api_raw")
RESPONSES_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("api_data")
DATA_DIR.mkdir(exist_ok=True)

now = datetime.now().strftime('%Y%m%d-%H%M')

DIVISIONS = {
    "RDF10": "Natural sciences",
    "RDF20-21": "Engineering and technology",
    "RDF30": "Medical, health and life sciences",
    "RDF40": "Agricultural and veterinary sciences",
    "RDF50": "Social sciences",
    "RDF60": "Humanities and the arts"
}

GROUPES = {
    "RDF101": "Mathematics and statistics",
    "RDF102": "Computer and information sciences",
    "RDF103": "Physical sciences",
    "RDF104": "Chemical sciences",
    "RDF105": "Earth and related environmental sciences",
    "RDF106": "Biological sciences",
    "RDF107": "Other natural sciences",
    "RDF201": "Civil engineering, maritime engineering, transport engineering, and mining engineering",
    "RDF202": "Industrial, systems and processes engineering",
    "RDF203": "Electrical engineering, computer engineering, and information engineering",
    "RDF204": "Mechanical engineering",
    "RDF205": "Chemical engineering",
    "RDF206": "Materials engineering and resources engineering",
    "RDF207": "Medical and biomedical engineering",
    "RDF208": "Environmental engineering and related engineering",
    "RDF209": "Environmental biotechnology",
    "RDF210": "Industrial biotechnology",
    "RDF211": "Nano-technology",
    "RDF212": "Other engineering and technologies",
    "RDF301": "Basic medicine and life sciences",
    "RDF302": "Clinical medicine",
    "RDF303": "Health sciences",
    "RDF304": "Medical biotechnology",
    "RDF305": "Other medical sciences",
    "RDF401": "Agriculture, forestry, and fisheries",
    "RDF402": "Animal and dairy sciences",
    "RDF403": "Veterinary sciences",
    "RDF404": "Agricultural biotechnology and food sciences",
    "RDF405": "Other agricultural sciences",
    "RDF501": "Psychology and cognitive sciences",
    "RDF502": "Economics and business administration",
    "RDF503": "Education",
    "RDF504": "Sociology and related studies",
    "RDF505": "Law and legal practice",
    "RDF506": "Political science and policy administration",
    "RDF507": "Social and economic geography",
    "RDF508": "Media and communications",
    "RDF509": "Other social sciences",
    "RDF601": "History, archaeology and related studies",
    "RDF602": "Languages and literature",
    "RDF603": "Philosophy",
    "RDF604": "Arts (arts, history of arts, performing arts, music), architecture and design",
    "RDF605": "Other humanities"
}

SYSTEM_PROMPT = f"""Classify research grants using the CRDC 2020 Field of Research classification.
Input: one grant per line as `id | title | committee`.
Titles of the grants are in English or French. Committee name is always English. For each grant, return the 3 most likely Divisions and the 5 most likely Groups, in order of most likely to least likely. Score your certainty for every classification as a float between 0 (random guess) and 1 (absolutely certain). Some grants don't have a committee; consider the committee as a hint instead of a constraint. Use only the provided information to make your decision. A code can only appear once per grant; use the next closest category instead of repeating. Structure your output in JSON according to the provided schema.

DIVISIONS
{chr(10).join(f"{code}: {div}" for code, div in DIVISIONS.items())}

GROUPS
{chr(10).join(f"{code}: {gr}" for code, gr in GROUPES.items())}
"""

# fonction pour structurer réponse

def scorer(
    codes: list
    ) -> dict:

    """
    Création de la structure d'un objet JSON
    à partir des codes du CRDC, en limitant
    les valeurs possibles.
    """

    return {"type": "array", "items": {
        "type": "object",
        "properties": {"code": {"type": "string", "enum": codes},
                       "score": {"type": "string", "minimum": 0, "maximum": 1}},
        "required": ["code", "score"], "additionalProperties": False
    }}


# schéma JSON

SCHEMA = {"type": "object", "properties": {"results": {"type": "array", "items": {
    "type": "object",
    "properties": {"id": {"type": "integer"},
    "div": scorer(list(DIVISIONS)),
    "gr": scorer(list(GROUPES))},
    "required": ["id", "div", "gr"], "additionalProperties": False}}},
    "required": ["results"], "additionalProperties": False}

# peut-être pas utile

def valider(
    donnees: dict,
    id_attendus: set
    ):

    """
    S'assure que tous les projets ont été classifiés
    par le modèle et que tous les codes sont uniques.
    """

    res = {datum["id"] for datum in donnees["results"]}
    assert res == id_attendus, "Mauvais ID"
    for d in donnees["results"]:
        assert len(d["div"]) == 3 and len({x["code"] for x in d["div"]}) == 3
        assert len(d["gr"]) == 5 and len({x["code"] for x in d["gr"]}) == 5

def classify(
    batch: pd.DataFrame,
    no_batch: int
    ) -> dict:

    """
    Envoie les requêtes à l'API de OpenAI en fonction
    des variables spécifiées en tête de programme.

    L'algorithme a un failsafe pour reprendre une
    batch qui aurait échouée à partir de fichiers
    JSON générés procéduralement et faisant office
    de cache, dans le dossier RESPONSES_DIR.

    Les titres et comités sont envoyés au modèle avec
    un identifiant unique numérique. Chaque réponse
    du modèle ne peut dépasser 8000 tokens et doit être
    structurée au format JSON (dict Python).

    En cas d'échec, le programme retente 4 fois avec
    un délai exponentiel.
    """

    cache = RESPONSES_DIR / f"batch_{no_batch:04d}.json"
    if cache.exists():
        return json.loads(cache.read_text())["data"]

    user_req = "\n".join(f"{idu} | {grant['title']} | {grant['committee']}" for idu, grant in batch.iterrows())

    for tentative in range(4):
        try:
            response = client.with_options(timeout=900.0).responses.create(
                model = MODEL,
                instructions = SYSTEM_PROMPT,
                input = user_req,
                service_tier = "flex", # seulement disponible avec Luna, algorithme devra être réécrit pour Sol ou Astra
                reasoning = {"effort": EFFORT},
                text = {
                    "format": {
                        "type": "json_schema",
                        "name": "crdc",
                        "strict": True,
                        "schema": SCHEMA
                        }
                    },
                max_output_tokens = 8000
            )

            data = json.loads(response.output_text)
            
            valider(data, set(batch.index))
            cache.write_text(json.dumps({
                "id_response": response.id,
                "model": response.model,
                "usage": response.usage.model_dump(),
                "input": user_req,
                "data": data
            }, ensure_ascii=False, indent=1))

            return data

        except Exception as e:
            print(f"Tentative {tentative + 1} de la batch {no_batch} a échoué : {e}")
            time.sleep(2 ** tentative)

    raise RuntimeError(f"Batch {no_batch} timed out")


def applatir(
    donnees: dict,
    titres: dict
    ) -> list[dict]:

    """
    Organise les résultats structurés obtenus
    par la fonction classify() dans une liste
    de dictionnaires.

    Associe les codes retournés par le modèle à
    leur description et l'ID unique du projet à
    son titre afin de structurer les résultats
    de manière identique aux résultats obtenus
    avec les modèles locaux.
    """

    rangees = []

    for projet in donnees["results"]:
        rangee = {"title": titres[projet["id"]]}
        for no, d in enumerate(projet["div"], 1):
            rangee[f"code_div_{no}"], rangee[f"label_div_{no}"], rangee[f"score_div_{no}"] = d["code"], DIVISIONS[d["code"]], d["score"]
        for no, d in enumerate(projet["gr"], 1):
            rangee[f"code_gr_{no}"], rangee[f"label_gr_{no}"], rangee[f"score_gr_{no}"] = d["code"], GROUPES[d["code"]], d["score"]
        rangees.append(rangee) # ou écrire ligne par ligne ?
    
    return rangees

if __name__ == "__main__":
    df = pd.read_csv("../data/projets_comites_complets-ENFR.csv", sep=";").reset_index(drop=True) # changer source données
    titres = df["title"].to_dict()

    rangees = []

    for b, start in enumerate(range(0, len(df), BATCH_SIZE)):
        rangees += applatir(classify(df.iloc[start:start+BATCH_SIZE], b), titres)

    Path(f"{DATA_DIR}/raw_{MODEL}_{now}.json").write_text(
        json.dumps(
            {"grants": rangees},
            ensure_ascii=False,
            indent=2
        )
    )
    pd.DataFrame(rangees).to_csv(f"{DATA_DIR}/raw_{MODEL}_{now}.csv", index=False)