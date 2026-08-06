# Methodologie

> A lire avant toute autre doc. Ce fichier definit ce qu'on cherche a prouver, comment on le
> prouve, et ce qui debloque l'investissement suivant.

## Le probleme

Reconstruire une piece mecanique photographiee en GLB, pour un usage **visualisation /
catalogue**. La barre est donc : silhouette juste sous tous les angles, proportions correctes,
texture presente sur toutes les faces. **Pas** de precision millimetrique.

Le mono-vue echoue de facon previsible : la face arriere, jamais observee, est hallucinee par le
prior generatif et souvent mal texturee. C'est exactement le defaut qui se voit sur un catalogue.

## Ce que fait deja Pixal3D

Pixal3D est pixel-aligne : la sortie vit dans le repere de la vue d'entree, pas dans un espace
canonique. Le papier decrit une extension multi-vues qui back-projette les features multi-echelles
de chaque vue dans un volume 3D partage et **les moyenne par voxel** — c'est de l'**early fusion**,
appliquee au conditionnement, avant la generation.

**Consequence directe : le baseline a battre n'est pas le mono-vue, c'est le multi-vues natif.**

## Hypotheses

- **H1** — Le multi-vues natif de Pixal3D (B1) bat significativement la meilleure vue seule (B0)
  sur les metriques P1, en particulier sur les faces non observees en mono-vue.
- **H2** — B1 atteint la barre « visualisation / catalogue » sans traitement supplementaire.
- **H3** *(en reserve)* — Une fusion **late** de N generations independantes, ponderee par une
  confiance par voxel et contrainte par une enveloppe visuelle, bat B1.

H1 et H2 se testent au Sprint 1. **H3 ne se teste que si H2 est refutee et si le diagnostic
d'erreur la rend plausible.**

## Pourquoi H3 est en reserve et non abandonnee

La late fusion garde deux avantages structurels sur l'early fusion :
- Moyenner les features *avant* generation peut noyer une evidence contradictoire ; le modele
  doit ensuite halluciner un objet globalement coherent.
- Elle seule permet d'appliquer une **contrainte geometrique dure** — l'enveloppe visuelle — qui
  peut *opposer un veto* a l'hallucination. L'early fusion ne peut structurellement pas le faire.
- Les deux se composent : on peut late-fuser N generations elles-memes early-fusees.

Mais elle coute cher, et une sortie de flow matching **n'est pas une probabilite calibree** : les
signaux de confiance devront etre *mesures* (AUROC contre l'erreur GT par voxel), pas supposes.
D'ou la mise en reserve. Voir ADR-0001.

## Metriques, calibrees sur l'usage catalogue

| Rang | Metrique | Pourquoi |
|---|---|---|
| **P1** | IoU de silhouette sur vues held-out, dont vues arriere | Le plus correle au rendu percu ; detecte directement la face arriere hallucinee |
| **P1** | LPIPS sur rendus de vues nouvelles | L'apparence compte autant que la forme pour un catalogue |
| P2 | F-score@tau, tau = 1 % de la diagonale de bbox | Proportions et forme discriminante |
| P3 | Chamfer bidirectionnel | Diagnostic fin, jamais un critere de decision |
| Diag | Erreur restreinte aux faces non observees en mono-vue | Isole precisement le gain attendu du multi-vues |

Toutes les metriques geometriques sont calculees **apres alignement Sim3** : l'echelle absolue
n'est pas un critere a ce stade (voir « Extension dimensionnelle » plus bas).

## Protocole

Benchmark **synthetique d'abord** : meshes CAO rendus sous N vues, avec poses et intrinseques
verite terrain. Cela **isole l'algorithme de l'erreur de pose**, qui est le maillon faible en
conditions reelles. Le passage aux photos reelles (poses estimees par VGGT) vient ensuite, et la
degradation synthetique -> reel est elle-meme une mesure.

Chaque run ecrit `runs/<ts>/` avec `manifest.json` (sha git, revision modele, seeds, params,
poses), `metrics.json`, `logs.jsonl` et une planche de rendus comparatifs. **Sans manifest, un
resultat est nul et non avenu** : il n'est pas reproductible, donc il ne prouve rien.

## Le gate F13

A l'issue de F12, une ADR dans `DECISIONS.md` doit statuer entre trois issues :

1. **B1 passe la barre catalogue** -> on saute entierement la recherche fusion, direction
   Sprint 3 (mise en prod). Issue la plus probable compte tenu de l'usage vise.
2. **B1 bat B0 mais reste sous la barre** -> **analyse d'erreur obligatoire avant tout code**.
   L'erreur est-elle *localisee* sur les faces peu observees et aux jonctions entre vues
   (-> Sprint 2 pertinent), ou *globale* : derive d'echelle, topologie fausse, texture absente
   (-> Sprint 2 inutile, il faut un autre levier — plus de vues, meilleure capture, fine-tuning
   domaine) ?
3. **B1 ne bat pas B0** -> le probleme est presque certainement dans les poses ou l'agregation,
   pas dans le modele. On debogue F10/F11 avant toute autre chose.

**Aucune feature de Sprint 2 ne peut passer a `active` avant que F13 soit `passing`.**

## Risques assumes

- **Le recalage est le maillon faible.** Des poses fausses font que le multi-vues *degrade* au
  lieu d'ameliorer. D'ou le gate de qualite de pose (F11) : sous le seuil, on refuse de fusionner
  plutot que de produire une reconstruction silencieusement degradee.
- **Les pieces mecaniques sont hors distribution** pour un prior entraine sur Objaverse : metal,
  peu texture, symetrique, brillant. Le prior hallucinera. A chiffrer, pas a supposer.
- **26 Go de poids** : le cold start serverless se compte en minutes, pas en 200 ms. L'API de
  production devra etre asynchrone (submit + polling).

## Extension dimensionnelle (Sprint 4, optionnel)

Passer de la fidelite geometrique a la fidelite dimensionnelle (mm) est un **ajout localise, pas
une refonte** : l'echelle absolue est un scalaire decouple de la forme. Tout le pipeline manipule
des **Sim3** (rotation, translation, echelle) et non des SE3, precisement pour reserver ce point
d'extension. Il suffira d'ajouter une calibration (mire ArUco ou etalon dans la photo) et une
metrique d'erreur en mm.
