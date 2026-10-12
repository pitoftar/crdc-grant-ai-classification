# Classification disciplinaire de projets financés

## Préambule

Le présent dépôt contient du code qui vise à effectuer une tâche de classification d'environ 2500 projets de recherche financés par l'un des trois grands conseils canadiens (CRSH, CRSNG et IRSC) dans le cadre d'un projet de recherche piloté par [Anton Boudreau Ninkov](https://ebsi.umontreal.ca/repertoire-ecole/professeurs/professeur/in/in34831/sg/Anton%20Boudreau%20Ninkov/).

Les projets doivent être classés selon les catégories du deuxième palier (groupes) de la [Classification canadienne de la recherche et développement (CCRD)](https://www150.statcan.gc.ca/n1/en/pub/89-26-0004/89260004-x2020001-eng.pdf) à partir de leur titre et de leur comité d'évaluation.

Une première tâche de classification avait été effectuée à l'été 2025 à l'aide de grands modèles de langage (LLM) en ligne (ChatGPT).
Cette méthode avait produit des résultats intéressants, mais avait été jugée améliorable en raison du manque de tracabilité et de reproductibilité de la sortie.

## Avertissement

Le code contenu dans ce dépôt a été écrit en Python par un programmeur autodidacte et très amateur.
Il contient très certainement des routines moins qu'optimales et des erreurs de logique en plusieurs endroits.
Son usage est par ailleurs limité à l'usage dans le cadre de ce projet en particulier.

Le projet aurait certainement été améliorable grâce à l'apport de LLM conversationnels en ligne (e.g. ChatGPT, Claude).
J'ai cependant décidé de le réaliser sans l'aide de ces systèmes pour plusieurs raisons.
D'abord, je souhaitais être en mesure de comprendre l'ensemble du code afin de le décrire et le défendre de manière appropriée dans les productions scientifiques (présentations et articles) en découlant éventuellement.
Ensuite, la compréhension profonde de la logique derrière le code et l'utilisation de structures algorithmiques et logiques que je maîtrise, même si elles ne constituent pas la manière la plus efficiente de réaliser la tâche, me permet de cibler plus naturellement la source des bogues et dysfonctionnements du code.
Finalement, si la programmation entièrement manuelle de cette section du code a certainement été plus chronophage qu'elle n'aurait pu l'être avec l'aide d'un LLM, je suis intimement convaincu que les aptitudes développées et la compréhension du code permettront de faciliter le processus de présentation des résultats et de lui permettre de gagner en robustesse.

## Utilisation

Le présent dépôt doit être cloné.
Il faut ensuite se déplacer dans le dépôt et exécuter le script `basic_pipeline.py`.

Le programme est contrôlé à l'aide du fichier de paramétrage intitulé `pipeline_config.py`.

## Choix méthodologiques

### Présupposés

L'expérience a été réalisée avec les présupposés suivants :
- Elle devait être la **plus reproductible possible** par souci de rigueur scientifique
- Elle devait être la **moins coûteuse possible** en argent et en ressources, afin de s'assurer de l'accessibilité de l'expérience (le plus grand nombre possible peut la reproduire)
- Elle devait produire une **sortie structurée ou semi-structurée** par souci pratique
- Il devrait être aisé pour une personne **sans aptitudes de programmation particulières** de prendre le projet en charge
- La classification doit s'opérer au niveau des groupes, mais les autres paliers de classification (division, classes et sous-classes) peuvent être utilisés pour orienter le classement

### Méthode

#### Classification sans exemple

Afin d'obtenir des résultats plus robustes, plus facilement défendables et plus aisément reproductibles qu'en faisant appel à des LLM en ligne.
En l'absence d'un corpus d'entraînement annoté portant sur les données utilisées pour la recherche, il fallait faire usage de grands modèles de langage (100M+ paramètres) spécialisés pour la « classification _zero-shot_ » (_zero-shot classificatin_, ZSC) ou « classification sans exemple ».

En raison de sa licence ouverte (au moment de rédiger ceci à l'automne 2026) et de l'accès privilégié qu'elle offre à une variété de modèles de langage, la bibliothèque `transformers` de HuggingFace s'est présentée comme une solution intéressante.
Cette librairie offre nativement une méthode de ZSC, basée sur des méthodes d'inférence du langage naturel (_natural language inference_, NLI), c'est-à-dire de déterminer la probabilité d'assentiment (_agreement_) entre une prémice et une hypothèse.
Dans la tâche courante, les titres des projets constituent la prémice et les classes du CCRD constituent autant d'hypothèses, par rapport auxquelles le LLM évalue l'assentiment.

La sortie est un fichier `csv` comportant la séquence évaluée (titre et comité, le cas échéant), les trois divisions jugées les plus probables de constituer une catégorisation adéquate pour le projet et le score d'assentiment, ainsi que les cinq groupes les plus probable et le score d'assentiment.

#### Raffinement des catégories

Plusieurs méthodes de raffinement maison des catégories ont été testées afin de tenter d'améliorer la précison des résultats.

##### Par comité

Certains projets financés ont été évalués par un comité disciplinaire.
Afin de potentiellement offrir une résonance avec les catégories disciplinaires du CCRD et puisque le titre d'un projet est souvent trop spécifique pour permettre de déduire naturellement et aisément la discipline à laquelle il appartient, les noms des comités ont été intégrés entre parenthèses aux titres des projets lorsqu'ils étaient significatifs.
La catégorisation des comités comme significatifs ou non-significatifs a été effectué manuellement par l'équipe de recherche en amont de la tâche de classification.

##### Par division

Par défaut, les projets étaient classifiés selon leur division et selon leur comité, indépendamment l'une de l'autre.
Autrement dit, le modèle examinait le titre du projet et vérifiait par NLI sa correspondance avec chacune des 6 divisions.
Il effectuait ensuite la même tâche avec chacun des 43 groupes.
La réflexion derrière ce choix était la suivante : devant des division aussi larges que « Social sciences » ou « Engineering and technology », il semblait possible que le modèle associe le titre d'un projet avec une catégorie générale qui semble convenir adéquatement en absence de plus de détails.
Il pourrait ensuite découvrir, parmi les 43 groupes, une catégorie correspondant mieux au projet d'après son titre.
Par exemple, un projet intitulé « Quelles sont les mathématiques utilisées par les ingénieurs au quotidien? Une étude sur les pratiques des ingénieurs pour déterminer leurs besoins mathématiques » et enrichi du comité « Education and social work » pourrait être catégorisé dans la division « Engineering and technology » par le modèle, en raison de la présence des termes « ingénieurs » et « mathématiques », mais pourrait être classé dans le groupe « Education », dont l'imbrication dans la division « Social sciences » n'est pas évidente si l'on considère seulement le premier niveau de la classification.

Une autre méthode a été développée en réduisant les choix de classification de groupe à ceux contenus dans la division identifiée comme la plus probablement.
Dans ce deuxième flux de travail, baptisé « classification limitée » (`ltd`), le modèle détermine la division la plus probablement pour chaque projet, et ne se considère que les groupes issus de cette division pour établir le groupe le plus pertinent.
Cette méthode ne présente en sortie qu'une seule division et trois groupes.

##### Enrichissement des catégories

Une troisième méthode, baptisée « classification raffinée » (`finet`), a été développée afin de « rendre visibles » les niveaux inférieurs du CCRD sans qu'ils ne soient directement l'objet de la tâche de classification.
Dans cette méthode, les étiquettes des groupes sont ajoutés à celles des divisions lors de la tâche de classification afin qu'elles soient considérées par le modèle au moment de déterminer l'assentiment entre le titre du projet et les catégories disciplinaires.
Par exemple, plutôt de considérer la catégorie « Engineering and technology », le modèle considère une catégorie qui se présente un peu comme suit : « Engineering and technology (includes Civil engineering, maritime engineering, transport engineering, and mining engineering, Industrial, systems and processes engineering, Electrical engineering, computer engineering, and information engineering ... ) ».
Similairement, les groupes sont enrichis des classes, qui constituent le niveau inférieur.

Une méthode utilisant la classification limitée par division et les catégories enrichies a également été mise en place.
Elle est identifiée comme la « classification raffinée limitée » (`ltd+finet`).

Une variante de cette méthode a également été testée en alimentant des divisions avec toutes leurs sous-classes.
Les étiquettes devenaient cepndant trop longues et dépassaient la fenêtre de contexte du modèle (variable, mais souvent autour de 512 tokens), ce qui a grandement réduit la précision de la tâche de classification.

#### Multilinguisme

Le travail est effectué sur les produits de la recherche d'une université québécoise (l'Université de Montréal), ce qui signifie que le jeu de données comporte un certain nombre de projets dont le titre est en français.
La plupart des modèles de langage produisent des résultats moins précis sur les corpus non-anglophones, d'où le choix de tester également des LLM multilingues sur le corpus.

Un [flux de travail expérimental](https://github.com/pitoftar/crdc-grant-ai-classification/blob/main/classification/fr_test_pipeline.py) permettant de classifier indépendamment les projets en anglais a été esquissé, mais non abouti.
Il pourrait éventuellement être implémenté.

### Modèles

Les [modèles utilisés](https://github.com/pitoftar/crdc-grant-ai-classification/blob/main/classification/models.txt) sont tous disponibles gratuitement sur HuggingFace au moment d'écrire ces lignes.
Ils ont tous été développés entre 2023 et 2025 et disposent d'entre 200M et 600M de paramètres.
Certains modèles sont unilingues anglais, alors que d'autres sont réputés multilingues.

Le programme est écrit d'une manière à permettre une grande interchangeabilité des modèles.

### Pile logicielle

À venir (un fichier `requirements.txt` doit être généré à partir du script de classification)

## Structure du dépôt

```
.
├── classification
│   ├── basic_pipeline.py
│   ├── fr_test_pipeline.py
│   ├── models.txt
│   ├── pipeline_config.py
│   └── snapshot.py
├── data
├── out
│   ├── div_multilabel_refined_w_subclass
│   │   ├── cross-encoder-nli-deberta-v3-base
│   │   ├── facebook-bart-large-mnli
│   │   ├── log
│   │   ├── MoritzLaurer-bge-m3-zeroshot-v2.0
│   │   ├── MoritzLaurer-DeBERTa-v3-base-mnli-fever-anli
│   │   ├── MoritzLaurer-deberta-v3-large-zeroshot-v2.0
│   │   ├── MoritzLaurer-mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
│   │   └── MoritzLaurer-ModernBERT-large-zeroshot-v2.0
│   ├── log
│   └── models
│       ├── cross-encoder-nli-deberta-v3-base
│       ├── facebook-bart-large-mnli
│       ├── MoritzLaurer-bge-m3-zeroshot-v2.0
│       ├── MoritzLaurer-DeBERTa-v3-base-mnli-fever-anli
│       ├── MoritzLaurer-deberta-v3-large-zeroshot-v2.0
│       ├── MoritzLaurer-mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
│       └── MoritzLaurer-ModernBERT-large-zeroshot-v2.0
└── results
    ├── experimental
    ├── finet
    ├── ltd
    ├── ltd+finet
    └── raw
```

Le dossier `out` contient tous les fichiers de sortie produits pendant le développemement du script.
Ils sont classés par modèle, puis par date, puis par la méthode, et finalement par la taille du jeu de données utilisé (petit échantillon, moyen échantillon, jeu de données complet).

Le dossier `results` ne contient que les fichiers qui correspondent à l'**itération la plus récente** de l'opération de classification.
Les résultats sont classés par méthode, puis par modèle.
Tous les fichiers de sortie présents dans ce dossier représentent le jeu de données complet.
**Il n'y a qu'un résultat par méthode par modèle** dans ce dossier.

Le sous-dossier `experimental` dans le dossier `results` correspond à une classification où les étiquettes disciplinaire sont enrichies à l'aide de leurs sous-classes, jugée moins précise.

## Problèmes connus

### Raffinement

- Le raffinement avec sous-classes semble, avant l'analyse des résultats, présenter une baisse de précision importante par rapport aux autres méthodes. Celle-ci est expliquée puisque les séquences composées de la division et de toutes les sous-classes sont trop longues et dépassent la fenêtre de contexte du modèle (souvent 512 tokens).
- À noter que le modèle entraîné par Facebook (BART Large MNLI) a une fenêtre de contexte de 1024 tokens et a probablement un peu mieux géré ces cas.
- Les divisions, qui comportent beaucoup plus de sous-classes par étiquette, sont probablement plus touchées par ce problème que les groupes.

### Paramétrage

- L'un des objectifs du projet était qu'il soit réutilisable par une personne sans qu'elle n'ait à toucher le code. Cet objectif n'est pas tout à fait atteint dans l'état actel des choses, puisque le paramétrage de raffinement des classes n'est pas possible sans changer les référencements dans le code. L'option `finet` produit actuellement les étiquettes suivantes « `div` (includes `gr`, `gr`, `gr`, `gr`... ) » et « `gr` (includes `cls`, `cls`, `cls`...) ». Il serait souhaitable être capable de déterminer facilement quel niveau doit être utilisé pour enrichir les classes à partir du fichier de paramétrage.
- pas possible d'ajuster le nombre de classification premier niveau ou 2e niveau

### Utilisation

- Un fichier `requirements.txt` reste à produire pour le script afin de faciliter l'installation des dépendances pour reproductibilité.

### Fichiers de sortie

- Lors de l'utilisation de la méthode expérimentale d'enrichissement des divisions et groupes par sous-classes, le fichier de sortie n'est pas propre et donne les catégories fournies au modèle plutôt que les catégories abrégées : i. e. « `div` (includes `scls`, `scls`, `scls`...) » plutôt que « `div` ». Cette erreur est probablement dûe à une mauvaise association entre les catégories données en entrée au modèle et le dictionnaire utilisé pour faire l'association entre celles-ci et les catégories à afficher en sortie.
- Les fichiers de sortie sont actuellement organisés selon le modèle utilisé pour la classification. Il ne s'agit peut-être pas de la méthode la plus pratique ou souhaitable pour la réutilisation des données. Une nouvelle organisation a été réalisée manuellement dans le dossier `results` pour faciliter l'utilisation des fichiers de sortie pour l'analyse.
- La production de la dernière version de la classification (dans le dossier `results`) n'est actuellement pas automatisée, alors qu'elle pourrait l'être.