# Familles de recettes — Clustering × Sorties structurées

Projet du module *Introduction à l'apprentissage automatique* (BUT 3), sujet n° 2.

Un LLM lit une recette écrite librement et en remplit une **fiche structurée** : ingrédients principaux, temps total, coût estimé, mode de cuisson. Un **K-Means** entraîné sur un millier de ces fiches regroupe les recettes en familles. Dans l'application, l'utilisateur colle une nouvelle recette et voit à quelle famille elle appartient, à quelle distance du centre elle se trouve et quelles recettes lui ressemblent le plus.

```
 recette en texte libre ──▶ LLM + schéma Pydantic ──▶ fiche JSON ──▶ vecteur ──▶ K-Means ──▶ famille
   (anglais ou français)     (sortie structurée)     {ingrédients,   (51 nombres)  (k centres)  + portrait
                                                      temps, coût,                               + recettes voisines
                                                      cuisson}
```

Chaque notion a un rôle indispensable. Sans le LLM, le K-Means n'a rien à regrouper : une recette en texte libre n'est pas un vecteur. Sans le K-Means, on obtient des fiches isolées, sans familles ni comparaison possible.

## 1. Installer et lancer

```bash
uv venv --python 3.12
source .venv/bin/activate          # Windows : .venv\Scripts\activate
uv pip install -r requirements.txt
cp .env.example .env               # puis mettre sa clé Mistral dans .env
```

Le fichier `.env` contient la clé d'API. Il est exclu du dépôt par `.gitignore`, et aucune clé n'apparaît dans le code. Un autre fournisseur compatible OpenAI (Groq, Ollama…) fonctionne aussi : il suffit de changer les trois lignes du `.env`.

Les sorties des étapes ci-dessous sont versionnées : l'échantillon (`data/`), les fiches extraites par le LLM, le modèle (`modeles/`) et les résultats (`resultats/`). Une fois le dépôt cloné, **`streamlit run app.py` suffit donc pour lancer l'application** ; la clé ne sert qu'à analyser une nouvelle recette. Pour tout refaire depuis le début :

| Étape | Commande | Durée |
|---|---|---|
| 1. Échantillon et texte libre | `python 1_preparer_donnees.py` (il faut `recipes.csv` à la racine, voir § 2) | 1 min |
| 2. Fiches par le LLM (zéro exemple) | `python 2_extraire_fiches.py` (tester d'abord avec `--limite 5`) | ~1 200 appels |
| 2 bis. Fiches few-shot pour l'expérience | `python 2_extraire_fiches.py --few-shot --split test --sortie data/fiches_llm_few_shot.jsonl` | ~220 appels |
| 3. Clustering et expériences | `python 3_entrainer_clustering.py` (puis `--k N` après avoir regardé `resultats/choix_k.png`) | 30 s |
| 4. Évaluation de l'extraction | `python 4_evaluer_extraction.py` | 5 s |
| Application | `streamlit run app.py` | — |

Le fichier produit par l'étape 2 sert de **cache** : une recette déjà traitée n'est jamais renvoyée au LLM, et on peut interrompre puis relancer le script. Le modèle est entraîné **hors de l'application** (étape 3, `joblib.dump`). L'application le charge une seule fois au démarrage (`st.cache_resource`) et ne le réentraîne jamais.

### Organisation du code

| Fichier | Rôle |
|---|---|
| `llm.py` | **IA générative** : schéma `FicheRecette` (Pydantic), prompt, exemples few-shot, fonction `appeler` du TP6 (pauses en cas d'erreur 429) |
| `features.py` | **IA prédictive, préparation** : `VectoriseurFiche`, un transformateur scikit-learn qui change une fiche en vecteur |
| `3_entrainer_clustering.py` | **IA prédictive** : choix de k, K-Means, comparaison des représentations, portraits des familles |
| `4_evaluer_extraction.py` | évaluation de la sortie structurée sur les recettes de test |
| `recettes.py`, `config.py` | lecture du format Food.com, chemins |
| `app.py` | application Streamlit |

## 2. Données

**[Food.com – Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)**, Kaggle (auteur : irkaal), licence **CC0 : domaine public**. Il contient 522 517 recettes en anglais publiées sur Food.com. Le fichier `recipes.csv` (700 Mo) n'est pas versionné : il faut le télécharger sur Kaggle et le placer à la racine.

**Notre échantillon** (`1_preparer_donnees.py`, graine 42) : 1 200 recettes, dont 981 pour l'entraînement et 219 pour le test. On garde les recettes :

- notées par au moins 3 personnes, pour écarter les recettes peu fiables ;
- dont la liste d'ingrédients est **complète**. Food.com perd souvent des ingrédients (la levure d'un pain, par exemple), et la liste des noms devient alors plus courte que celle des quantités. On ne garde que les recettes où les deux listes sont alignées, soit une sur quatre ;
- qui appartiennent aux 20 catégories de plats les plus fréquentes. Les pseudo-catégories de durée comme « < 30 Mins » sont écartées.

**Le texte envoyé au LLM** contient le nom, les ingrédients avec leurs quantités et les étapes. On n'y met **ni la durée, ni la catégorie, ni les mots-clés** : le LLM doit les déduire, comme pour une recette collée par un utilisateur. Ces champs sont gardés à part et servent de **vérité terrain** pour évaluer l'extraction.

**Données rédigées par nous** :

- les deux recettes d'exemple du prompt few-shot (`llm.py`). Elles sont inventées, pour qu'aucune recette évaluée ne serve d'exemple ;
- `data/annotations_cout.csv` : 40 recettes de test dont **le coût est annoté à la main**. Food.com ne donne aucune vérité terrain pour le coût. 🛠️ *Colonne `cout_reference` à remplir (faible / moyen / élevé).*

## 3. IA générative : la sortie structurée

### Le schéma

```python
class FicheRecette(BaseModel):
    ingredients_principaux: list[str]   # 3 à 6, nom générique anglais au singulier : "chicken", "flour"
    temps_total_minutes: int            # préparation + cuisson + temps passifs (repos, levée, marinade)
    cout_estime: Literal["faible", "moyen", "élevé"]
    mode_cuisson: Literal["four", "plaque de cuisson", "grill ou barbecue", "friture",
                          "mijoteuse ou autocuiseur", "micro-ondes", "sans cuisson"]
```

Chaque champ porte une `description`, que le modèle lit. Les appels se font à **température 0** : on veut une extraction stable, pas de créativité. La recette est placée entre balises `<recette>…</recette>`, et le message `system` précise que ce texte est une donnée, ce qui protège contre une injection de prompt.

**Pourquoi ces choix :**

- **`Literal` pour le coût et le mode de cuisson.** Le modèle ne peut répondre qu'une des valeurs prévues, ce qui donne directement des colonnes one-hot. Avec un texte libre, on obtiendrait « oven », « baked », « au four » : trois valeurs pour une seule idée.
- **Des ingrédients génériques, en anglais.** Le clustering compare des ingrédients : « boneless skinless chicken breasts » et « chicken thighs » doivent tous les deux donner `chicken`. C'est le LLM qui fait cette normalisation, que des règles écrites à la main feraient mal. On impose l'anglais pour que le vocabulaire reste le même quand l'utilisateur colle une recette en français.
- **Le coût en trois classes plutôt qu'en euros.** Le LLM n'a pas les prix : un montant serait une hallucination précise. Un ordre de grandeur est plus honnête.
- **Le format est garanti, pas le contenu** (TP6) : d'où l'évaluation champ par champ ci-dessous. Après l'extraction, une petite normalisation déterministe passe les noms en minuscules, supprime les doublons et garde au plus 6 ingrédients.

### L'expérience : zéro exemple ou few-shot ?

On compare deux prompts sur **les mêmes recettes de test**, celles qui n'ont servi ni à écrire le prompt ni à entraîner le clustering. Le prompt « zéro exemple » contient seulement les consignes. Le prompt « few-shot » ajoute deux recettes résolues (des tours `user` / `assistant`). Chaque champ est comparé à une vérité terrain :

| Champ | Vérité terrain | Métrique |
|---|---|---|
| temps | `TotalTime` de Food.com | erreur absolue médiane (min) ; part des recettes à ±25 % ; part sous-estimée de plus de moitié |
| mode de cuisson | mots-clés Food.com « Oven », « Stove Top », « No Cook »… (quand un seul est présent) | exactitude, matrice de confusion |
| ingrédients | liste complète des ingrédients de la recette | précision : part des ingrédients extraits qui figurent vraiment dans la recette (les autres sont **inventés**) |
| coût | annotations à la main ; mot-clé « Inexpensive » (indice faible) | exactitude ; part de « faible » avec et sans le mot-clé |

**Résultats** (`resultats/extraction_comparaison.csv`, modèle `…`, date `…`) :

| | zéro exemple | few-shot |
|---|---|---|
| temps : erreur absolue médiane | … | … |
| temps : part à ±25 % | … | … |
| mode de cuisson : exactitude | … | … |
| ingrédients : précision | … | … |
| coût : exactitude (annotations) | … | … |
| tokens envoyés par recette | … | … |

*Interprétation : à rédiger après les mesures. Points à regarder : les plus grosses erreurs de temps, affichées par le script, sont-elles des temps passifs oubliés (« refrigerate overnight ») ? Le few-shot vaut-il les tokens supplémentaires, c'est-à-dire son coût ? Avec 219 recettes de test, un écart de 2 ou 3 points n'est pas significatif.*

## 4. IA prédictive : le clustering

### De la fiche au vecteur (`features.py`)

| Variables | Codage | Pourquoi |
|---|---|---|
| temps | `log(1 + minutes)`, standardisé | ce sont les rapports qui comptent : 10 min contre 40 min est un écart aussi grand que 1 h contre 4 h. Sans le log, quelques recettes de 24 h écraseraient tout |
| coût | 0 / 1 / 2, standardisé | les trois classes sont ordonnées |
| mode de cuisson | 7 colonnes 0/1 (one-hot) | variable catégorielle sans ordre |
| ingrédients | une colonne 0/1 par ingrédient présent dans au moins 2 % des fiches d'entraînement (multi-hot) | le vocabulaire est appris **sur l'entraînement seulement**. Un ingrédient plus rare est ignoré, et l'application le signale |

**Les colonnes 0/1 ne sont pas standardisées.** C'est le piège des échelles du TP1, dans l'autre sens : un ingrédient présent dans 2 % des recettes vaudrait environ 7 après standardisation, et deux recettes qui le partagent paraîtraient très proches pour cette seule raison. En 0/1, chaque ingrédient différent ajoute 1 au carré de la distance. L'expérience B ci-dessous mesure l'effet de ce choix.

### Pourquoi K-Means

Il y a environ mille individus et une cinquantaine de variables, et on veut un **centre** par famille pour faire un portrait et affecter une nouvelle recette au centre le plus proche (`predict`). K-Means fait exactement cela. La CAH n'a pas de `predict` naturel. DBSCAN ne marche pas bien en dimension 50 avec des variables binaires, et il refuse d'affecter certaines recettes, alors que l'application doit toujours répondre. En revanche, K-Means range **toutes** les recettes, même celles qui ne ressemblent à aucune famille. L'application compense en signalant une **recette atypique** quand sa distance au centre dépasse celle de 95 % des recettes d'entraînement.

### Expérience 1 : le choix de k

On utilise le coude et la silhouette pour k de 2 à 12, sur l'entraînement (`resultats/choix_k.png`). Par défaut, le script retient le k de meilleure silhouette entre 4 et 10, et `--k` permet d'en imposer un autre. On ne prend pas k = 2 même si sa silhouette est meilleure : « sucré / salé » n'apprend rien à personne.

*Résultats et choix de k : à rédiger (valeurs de silhouette, forme du coude, raison métier du choix).*

### Expérience 2 : quelle représentation des recettes ?

Un clustering n'a pas de bonne réponse. On peut quand même mesurer si ses familles **retrouvent une structure connue** que le modèle n'a jamais vue : les catégories Food.com (« Dessert », « Breads », « Beverages »…). On fixe le même k, on entraîne sur l'entraînement, on affecte les recettes de **test** avec `predict`, puis on mesure l'accord avec leur catégorie :

- **AMI** (*adjusted mutual information*) et **ARI** (*adjusted Rand index*) valent environ 0 pour un découpage au hasard et 1 pour un accord parfait. On prend les versions **ajustées** : sur 219 recettes, la NMI d'un découpage tiré au hasard vaut déjà environ 0,07 ;
- la **silhouette** de chaque représentation est donnée à titre indicatif. Elle **ne se compare pas** d'une ligne à l'autre, car les distances sont calculées dans des espaces différents.

| Représentation | AMI test | ARI test |
|---|---|---|
| A. fiches LLM, colonnes 0/1 non standardisées (modèle retenu) | … | … |
| B. fiches LLM, tout standardisé | … | … |
| C. sans LLM : TF-IDF de la liste complète des ingrédients Food.com | … | … |
| D. familles tirées au hasard (plancher) | ≈ 0 | ≈ 0 |

La ligne C est la **référence sans IA générative**. Si elle fait aussi bien que A, le LLM n'apporte rien au clustering lui-même. Il reste alors utile pour deux choses que C ne sait pas faire : traiter une recette en texte libre sans liste d'ingrédients propre, et apporter le temps, le coût et le mode de cuisson.

Le script mesure aussi la **stabilité** : l'ARI moyen entre 5 K-Means lancés avec des graines différentes (`n_init=1`). Un ARI proche de 1 signifie que les familles ne dépendent pas du hasard de l'initialisation.

*Résultats : à recopier depuis `resultats/comparaison_representations.csv` et à interpréter.*

### Les familles

*À compléter : un tableau avec, pour chaque famille, son nom (persona), son effectif, son temps médian, ses ingrédients typiques et ses catégories Food.com dominantes (voir `modeles/familles.json`). Les noms automatiques, de la forme « ingrédients typiques — mode de cuisson », peuvent être remplacés par des noms choisis à la main dans `noms_familles.json` (`{"0": "Pâtisseries au four", …}`), que l'application lit au démarrage.*

## 5. L'application

- **Classer une recette** : on colle une recette ou on en tire une au hasard parmi les recettes de test. L'application affiche la fiche JSON produite par le LLM et les tokens consommés, puis la famille, la distance à chaque centre (une recette à la frontière hésite entre deux familles), les 5 recettes d'entraînement les plus proches et la position de la recette sur l'ACP.
- **✏️ Modifier la fiche à la main** : on change le mode de cuisson, le temps ou un ingrédient, et on voit si la famille change. Cela sert pendant l'exposé (« à votre avis, si on fait cette recette sans cuisson… ? ») et montre quelles variables pèsent dans la distance.
- **Les familles** : la projection ACP, le portrait de chaque famille et les centroïdes.
- **Expériences** : les figures et tableaux de `resultats/`.

## 6. Limites constatées

*À compléter avec ce que montrent vos mesures. Limites déjà connues :*

- **La vérité terrain est imparfaite.** `TotalTime` est saisi par l'auteur de la recette, parfois de façon fantaisiste. Les mots-clés de cuisson ne sont présents que pour environ 30 % des recettes. Le mot-clé « Inexpensive » n'indique qu'une tendance.
- **Les résultats du LLM ne sont pas parfaitement reproductibles**, même à température 0. Le nom `ministral-8b-latest` désigne un modèle que le fournisseur met à jour. Il faut noter le modèle et la date de chaque mesure.
- **Le vocabulaire d'ingrédients est figé à l'entraînement.** Un ingrédient absent (« jicama », « tofu » s'il est rare dans l'échantillon) est ignoré par le modèle. Une recette exotique tombe alors dans la famille dont les ingrédients connus sont les plus proches.
- **K-Means suppose des familles « rondes » et de tailles comparables**, et range toute recette dans une famille, d'où l'indicateur « atypique ».
- **L'ACP ne conserve qu'une partie de la variance** : deux familles qui se chevauchent sur le dessin peuvent être bien séparées dans l'espace complet.
- **Le coût est subjectif** et dépend du pays et de l'époque. Le LLM l'estime sans connaître aucun prix.
- **Coût des appels** : environ 1 200 appels pour construire le jeu, et un appel par recette dans l'application. Avec plus de recettes, il faudrait paralléliser les appels ou utiliser un modèle local (Ollama).
