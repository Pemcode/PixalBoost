# Tests

> A lire avant d'ecrire le moindre test.

## Le probleme du TDD sur un projet ML

Le red-green-refactor classique suppose une sortie deterministe et une assertion d'egalite. Un
modele generatif n'offre ni l'une ni l'autre. Appliquer le TDD naivement partout produit soit des
tests qui ne testent rien (`assert result is not None`), soit des tests instables.

D'ou **deux regimes**, avec une frontiere qui est aussi une frontiere de paquet.

## Regime deterministe — `src/pixaboost/core/`

**Red-green-refactor strict, sans exception.** Le test est ecrit, execute, et **constate rouge**
avant la premiere ligne d'implementation. Si tu n'as pas vu le test echouer, tu ne sais pas ce
qu'il teste.

Les fixtures sont **analytiques, a reponse fermee**. Exemples du type d'assertion attendu :

- un Sim3 compose de son inverse donne l'identite, a la tolerance numerique pres ;
- l'IoU de silhouette d'un mesh contre lui-meme vaut exactement 1.0 ;
- le Chamfer d'un nuage contre lui-meme vaut 0.0 ;
- le F-score d'une sphere analytique contre son maillage marching-cubes tend vers 1.0 quand la
  resolution augmente ;
- projeter puis back-projeter un point a sa profondeur connue redonne le point de depart.

Aucun fichier binaire, aucun telechargement, aucune sortie de modele. Ces tests doivent tourner
hors ligne, en moins de 60 secondes cumulees, et etre reproductibles bit a bit.

## Regime stochastique — `src/pixaboost/backends/` et sorties de modele

Pas d'assertion d'egalite. Deux outils a la place :

**Contract tests.** On enregistre un artefact reel (un GLB, un volume) une fois, on le commite ou
on le met en cache, et on teste que l'adaptateur sait toujours le lire et le convertir vers le bon
type de `core/`. Ce test detecte une rupture d'interface, pas une regression de qualite.

**Gates metriques.** La qualite se mesure, elle ne s'asserte pas. Les seuils vivent dans
`bench/thresholds.json`, sont versionnes, et **ne peuvent que monter**. Un seuil qu'on baisse pour
faire passer la CI est un mensonge inscrit dans le depot.

Pour une feature de recherche, le « rouge » c'est donc : **ecrire la metrique et l'assertion de
seuil qui echoue, avant l'algorithme.**

## Decoupage des repertoires

| Repertoire | GPU | Reseau | Dans `poe check` | Marqueur |
|---|---|---|---|---|
| `tests/unit/` | non | non | oui | aucun |
| `tests/integration/` | non | non | oui | aucun |
| `tests/e2e/` | oui | oui | **non** | `@pytest.mark.gpu` ou `@pytest.mark.network` |

`tests/integration/` travaille sur des **artefacts en cache** : c'est ce qui permet d'iterer sur
la metrologie sans GPU local. Un test d'integration qui declenche une inference est mal place.

## Pourquoi `core/` ne doit importer ni torch ni reseau

C'est ce qui garantit un gate de moins de 60 secondes, executable hors ligne et en CI gratuite.
La contrainte est verifiee par `tests/unit/test_architecture.py`, qui echoue si un module de
`core/` importe `torch`, `requests`, `socket`, ou equivalent.

Corollaire : LPIPS, qui est un reseau de neurones, vit dans `backends/perceptual.py` et non dans
`core/metrics.py`. Voir ADR-0003.
