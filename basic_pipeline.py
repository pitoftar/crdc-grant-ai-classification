from transformers import pipeline
import pandas as pd

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
df = pd.read_csv('data/crdc-full-encoder.csv', names=['single_encoder', 'code_d', 'code_g', 'code_c', 'code_sc', 'division' ,'group', 'class', 'subclass'])

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
dtfrm = pd.read_csv("data/sample.csv", sep=';', names=['comite_en', 'comite_fr', 'titre'])
projets = dtfrm.T.to_dict().values()
k = 'titre'
titres = [projet.get(k) for projet in projets if k in projet]

for titre in titres:
    resultat = classifier(titre, list(divisions.values()), multi_label=False)
    print(resultat)