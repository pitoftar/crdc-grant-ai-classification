from transformers import pipeline
import pandas as pd
import csv

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
df = pd.read_csv('data/crdc-full-encoder.csv', names=['single_encoder','code_d', 'code_g', 'code_c', 'code_sc', 'division' ,'group', 'class', 'subclass'])

# enlever la premiere ligne
df.drop(index=df.index[0], axis=0, inplace=True)

# stocker les divisions et groupes uniques dans un dictionnaire
def codes_uniques(dataframe, index, colonne):
    return dataframe.set_index(index)[colonne].to_dict()

divisions = codes_uniques(df, 'code_d', 'division')
groupes = codes_uniques(df, 'code_g', 'group')
classes = codes_uniques(df, 'code_c', 'class')
sous_classes = codes_uniques(df, 'code_sc', 'subclass')

# donnees projets
# initialiser liste des titres seuls
# initialiser liste des titres avec comites entre parenthese
# (le cas echeant)
dtfrm = pd.read_csv("data/smaller_sample.csv", sep=';', names=['comite_en', 'comite_fr', 'titre'])

projets = list(dtfrm.T.to_dict().values())
k = 'titre'
c = 'comite_en'

titres = [projet.get(k) for projet in projets if k in projet]

titres_comites = []

for projet in projets:
    if projet[c] is None:
        titres_comites.append(projet[k])
    else:
        titres_comites.append(f'{projet[k]} ({projet[c]})')

# classification des projets selon la division

resultats = []

for i, titre in enumerate(titres_comites):
    resultat = classifier(titre, list(divisions.values()), multi_label=False) # multilabel false pour la classification au niveau de la division
    resultats.append(resultat)
    print(f"Grant #{i+1} DONE")

output_net = []

for resultat in resultats:
    rangee = {}
    for k, v in resultat.items():
        if isinstance(v, list):
            for idx, val in enumerate(v, start=1):
                rangee.update({f"{k} {idx}": val})
        else:
            rangee.update({k : v})
    output_net.append(rangee)

with open('out/dump.txt', 'w') as f:
    f.write(str(output_net))
    f.close()

classification_division = pd.DataFrame(output_net)
print(classification_division)

# dump = pd.DataFrame.from_dict(resultats)
# dump.to_csv('out/facebook-bart-large-mnli.csv', sep=';', mode='w', quotechar='"') # ajuster le titre en fonction du traitement de la classification