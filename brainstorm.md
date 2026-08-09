## Hypotheses generales

Acuite plus basse sans les comites.

## Organisation generale des fichiers

Je n'ai pas actuellement mes documents Obsidian ou j'avais precisement documente ceci, mais je me demande comment organiser mes fichiers pour effectuer les diverses taches. J'ai plusieurs versions d'un meme programme qui font toutes a peu pres la meme chose, mais pas exactement.

Je pense que la maniere de faire preferable serait de declarer plusieurs fonctions pour effectuer toutes ces taches.
Par contre, je ne suis pas certain d'ou les inscrire: dans le meme programme ou dans d'autres programmes externes que j'appelle depuis le premier? Je ne connais pas les bonnes pratiques pythonesques, il faudrait que je me renseigne la-dessus.
Actuellement, je trouve que mon script est en train de devenir plutot long et de moins en moins lisible.

## Classification par groupes

En plus de la classification par divisions, il faut aussi operer une classification sur la base du groupe (43 choix).

Cette tache comporte certaines limitations :

- Il est deraisonable d'esperer classifier tous les projets selon chacun des 43 groupes. Il faudrait limiter l'output aux 5 ou 6 categories les plus probables.
- Dans un premier temps, on classifie les projets d'apres le groupe seul. Cependant, il faudra eventuellement trouver un moyen de tester la classification a partir de la division et d'operer une classification plus restreinte pour evaluer la performance comparativement au groupe seul.
- Il faut trouver un moyen de faire communiquer le dictionnaire de donnees qui contient les classifications par division et le dictionnaire de donnees qui contient les classifications par groupe. Il serait peut-etre preferable de travailler avec des dataframes pour manipuler ces donnees et effectuer ces taches, mais je serais plus a l'aise de l'implementer avec des dictionnaires pour l'instant.

### Limiter la classification

- Dans tous les cas, il faut faire la classification au complet. C'est au moment de l'affichage des resultats que l'on limite les resultats affiches dans le csv
- Utiliser `liste[:5]` sur la clef `labels` et les inscrire de maniere iterative dans le csv

### Division et groupe

C'est peut-etre l'etape la plus simple.

- Operer la classification selon la division comme d'habitude
    - Les resultats sont stockes dans une liste de dictionnaire (`resultats_division`) mais ne contiennent _que le string_ et pas le code
    - Le jeu de donnees `crdc-full-encoder` est deja charge dans un dataframe (`crdc`)
- Creer un sous-ensemble de groupes en fonction de leur classification
    - Segmenter le dataframe d'apres la valeur de la colonne ~~`code_d` ou~~ `division`
    - Transformer chacun des df en dictionnaire
    - Faire un check pour determiner la valeur de l'item de la liste de la clef `labels` a la position 1
    - Attribuer un dictionnaire de groupes a sonder en fonction de ce resultat

🦆 : La fonction `grouby` de pandas permet de generer un dictionnaire de dataframes en effectuant un regroupement (SQL-like) sur la base des valeurs d'une colonne en particulier.
J'ai stocke ces six differents dataframes dans un dictionnaire (`crdc_div`). Cependant, quand j'imprime chaque valeur du dictionnaire dans un for loop, j'obtiens seulement la clef et non la valeur (le dataframe).

🦆 : Je suis un peu coince avec le dictionnaire de dfs et je ne sais pas comment le gerer. Je dois ecrire une fonction qui
1) verifie la valeur de l'element a la position 0 dans la liste de resultats presentes sous la clef 'labels' pour chaque dictionnaire
2) verifie dans quel dataframe cette valeur est presente
3) fait une liste des groupes de ce dataframe
4) integre cette liste a la fonction `classifier()`

Ca bloque a l'etape 2, parce que les dataframes sont agreges dans un dictionnaire de dfs ou la clef est le label textuel et la valeur est le df en tant que tel.
Il faudrait probablement revoir cette methode de division des dfs. Actuellement, j'utilise la fonction `groupby()` pour diviser le dataframe. Qu'est-ce que je veux? Une liste de dicts? Un dict de listes de dicts?
Ce que je veux, c'est une liste des valeurs uniques dans une certaine colonne de chacun des df dans le dict. Je dois donc appliquer la meme operation a tous les dfs du dict.

🦆 : J'ai reussi a come up avec un loop qui fonctionne pour faire les etapes 1 a 4. Le probleme, maintenant, c'est que je n'arrive pas a me debarasser des nan, ni dans les noms de comites, ni dans les divisions, donc certains projets sont evalues comme 'nan'.
J'ai essaye plusieurs methodes a l'interieur de la boucle :

```
for k in dictionnaire:
    valeur = dictionnaire[k]
    valeur = [valeur for valeur in dictionnaire if str(valeur) != 'nan']
```

```
for k in dictionnaire:
    valeur = dictionnaire[k]
    valeur = [valeur for valeur in dictionnaire if valeur is not None]
```

```
for k in dictionnaire:
    valeur = dictionnaire[k]
    valeur = [valeur for valeur in dictionnaire if not valeur != valeur]
```

Rien ne marche.
J'ai aussi essaye plusieurs methodes pour retirer les nan des valeurs du dictionnaire apres son passage dans la fonction. J'ai par exemple essaye de le convertir en df avec `groupes_par_div = pd.DataFrame(groupes_par_div).dropna().to_dict('list')`, mais ca lance une erreur selon laquelle `All arrays must be of the same length`. Je ne peux donc pas repasser par un df.
Je ne suis pas certain du type de nan. Je pense que je devrais les gerer en amont, dans pandas, avec le df, parce que ce ne sont pas tout a fait les valeurs `None` natives a Python.

J'ai essaye d'appliquer `crdc_div = {code: discipline for code, discipline in crdc.dropna().groupby('division')}` au moment de creer le dictionnaire de valeurs par colonne.
Ca fonctionne.

### Integrer division a groupe

Combien de codes de division est-ce qu'on veut garder? On pourrait evidemment penser a un threshold a partir duquel on en garde plus d'un, mais ca compliquerait les choses pour l'affichage, et il ne me semble pas strictement necessaire d'inclure une boucle la ou on n'en a pas vraiment besoin.
On peut commencer par 3 et limiter ensuite au besoin.
Pour les groupes, on peut commencer par 5 et etendre ensuite au besoin.

Output souhaite:

|Idx|sequence|code_d_1@3|labels_d_1@3|scores_d_1@3|code_g_1@5|labels_d_1@5|scores_d_1@5|
|---|--------|----------|------------|------------|----------|------------|------------|
|0  |titre   |RDF10     |Natural sciences|0.9     |RDF101    |Mathematics |0.8         |

Probablement l'etape la plus compliquee

Les colonnes 1 a 5 dans le tableau sont deja le resultat du dechargement de la fonction `classifier()` dans une liste de dictionnaires, qui sont traitees ensuite dans un dataframe.
Il faudrait que je voie la maniere dont c'est gere : est-ce que l'association d'une paire k:v dans le dictionnaire division avec la meme paire k:v dans le dictionnaire groupe concatene les deux? Si oui, il y aurait peut-etre moyen de offload le dictionnaire de resultats de classification par division dans le dictionnaire de classification par groupes avec la methode {**dict}.