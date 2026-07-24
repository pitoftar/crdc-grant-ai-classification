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

# Build enriched Group labels from FOR structure
group_labels = {
    # "RDF301": "Basic medicine and life sciences — includes immunology, cancer biology, "
    #           "medical microbiology, neuroscience, pharmacology, physiology",
    # "RDF302": "Clinical medicine — includes cardiology, surgery, oncology, dermatology, "
    #           "diabetes, pediatrics, dentistry, psychiatry, radiology",
    # "RDF303": "Health sciences — includes nutrition, public health, epidemiology, "
    #           "rehabilitation, health services, nursing, care",
    # ... all 43 groups
}

# def classify_hierarchical(title, committee=None):
#     input_text = title if not committee else f"{title} [Committee: {committee}]"
    
#     # Stage 1: Division (6 candidates)
#     division_result = classifier(input_text, list(division_labels.values()), multi_label=False)
#     top_division = division_result["labels"][0]
    
#     # Stage 2: Group within that Division only
#     candidate_groups = {k: v for k, v in group_labels.items() 
#                         if k.startswith(division_map[top_division])}
#     group_result = classifier(input_text, list(candidate_groups.values()), multi_label=False)
    
#     top_group_label = group_result["labels"][0]
#     top_group_score = group_result["scores"][0]
    
#     return {
#         "division": top_division,
#         "group": top_group_label,
#         "confidence": top_group_score,
#         "flag_for_review": top_group_score < 0.6  # tune this threshold
#     }