## Classification par groupes

En plus de la classification par divisions, il faut aussi operer une classification sur la base du groupe (43 choix).

Cette tache comporte certaines limitations :

- Il est deraisonable d'esperer classifier tous les projets selon chacun des 43 groupes. Il faudrait limiter l'output aux 5 ou 6 categories les plus probables.
- Dans un premier temps, on classifie les projets d'apres le groupe seul. Cependant, il faudra eventuellement trouver un moyen de tester la classification a partir de la division et d'operer une classification plus restreinte pour evaluer la performance comparativement au groupe seul.
- Il faut trouver un moyen de faire communiquer le dictionnaire de donnees qui contient les classifications par division et le dictionnaire de donnees qui contient les classifications par groupe. Il serait peut-etre preferable de travailler avec des dataframes pour manipuler ces donnees et effectuer ces taches, mais je suis plus a l'aise de l'implementer avec des dictionnaires pour l'instant.

### Limiter la classification

- Dans tous les cas, il faut faire la classification au complet. C'est au moment de l'affichage des resultats que l'on limite
- Utiliser 

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


### Integrer division a groupe

Combien de codes de division est-ce qu'on veut garder? On pourrait evidemment penser a un threshold a partir duquel on en garde plus d'un, mais ca compliquerait les choses pour l'affichage, et il ne me semble pas strictement necessaire d'inclure une boucle la ou on n'en a pas vraiment besoin.
On peut commencer par 3 et limiter ensuite.

Output souhaite:

|Idx|sequence|code_d_1@3|labels_d_1@3|scores_d_1@3|code_g_1@5|labels_d_1@5|scores_d_1@5|
|---|--------|----------|------------|------------|----------|------------|------------|
|0  |titre   |RDF10     |Natural sciences|0.9     |RDF101    |Mathematics |0.8         |