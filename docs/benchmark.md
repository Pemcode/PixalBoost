# Benchmark synthetique

> A lire avant de toucher a `bench/` ou aux metriques.
> Construire : `uv run poe bench-build` (~5 s, 3 pieces x 18 vues en 512).

## Pourquoi synthetique d'abord

Le benchmark existe pour **isoler l'algorithme de l'erreur de pose**. En conditions reelles, une
reconstruction ratee peut venir du modele *ou* d'un recalage faux, et on ne sait pas laquelle.
Ici les poses sont exactes par construction, donc tout ecart mesure est imputable au modele.

Il est aussi **hors ligne, deterministe et gratuit** : pas de telechargement de dataset, pas de
GPU. C'est ce qui permet d'en faire un gate automatise.

## Protocole de prise de vue

**6 azimuts x 3 elevations (+45 deg, 0 deg, -45 deg) = 18 vues.**

C'est exactement le protocole des photos reelles de l'utilisateur. Ce n'est pas un detail :
si les deux protocoles differaient, la comparaison synthetique -> reel melangerait l'ecart de
domaine et l'ecart de geometrie de capture, et ne mesurerait plus rien d'interpretable.

La vue `az000_el+00` reproduit **exactement** la camera de conditionnement canonique de Pixal3D
(`front_view_camera`, voir `docs/pixal3d-internals.md`). Un test le verifie. Sans cela, toutes
les poses fournies au modele seraient silencieusement decalees par rapport au repere dans lequel
il a ete entraine.

## Les pieces

Procedurales, pas telechargees : le benchmark doit se construire hors ligne et se reproduire bit
a bit. De vraies pieces CAO pourront etre ajoutees derriere la meme interface `(vertices, faces)`.

| Piece | Topologie | Ce qu'elle teste |
|---|---|---|
| `l_bracket` | Prismatique, profil concave | Aretes vives, faces planes, concavite |
| `stepped_shaft` | Axisymetrique pleine | Surfaces courbes, trois diametres, symetrie de revolution |
| `flange_ring` | Genre 1 (percage traversant) | Un trou dans la silhouette sous la plupart des angles |

Toutes sont **etanches par construction** (extrusion capee, revolution refermee sur elle-meme) et
normalisees dans la boite `[-0.5, 0.5]` que Pixal3D utilise a l'export.

> **L'etancheite n'est pas cosmetique.** Un maillage a faces internes — ce qu'on obtient en
> concatenant naivement deux boites qui se recouvrent — place des points d'echantillonnage de
> reference **a l'interieur** du solide et corrompt silencieusement tous les Chamfer et F-scores
> en aval. Un test verifie que chaque arete est partagee par exactement deux faces.

## Rendu

Lambertien mat, lumiere dans l'axe camera, gris neutre, **fond noir**. Deux raisons : c'est ce
vers quoi Pixal3D pretraite les photos reelles (`preprocess_image`, fond noir), et c'est une
approximation honnete des pieces metalliques **mates** du jeu reel. Les normales sont utilisees
a plat, sans lissage : ce sont des pieces usinees, et l'ombrage facette garde lisibles les aretes
vives qui portent la forme.

Le rasteriseur est en numpy pur (`core/render.py`), z-buffer avec interpolation de profondeur
perspective-correcte. Pas de GPU, pas de contexte GL headless : c'est la condition pour que la
metrique P1 reste dans le gate.

## Ce que produit `bench-build`

```
data/bench/
├── manifest.json                 sha git, version, config, liste des pieces, horodatage UTC
└── <piece>/
    ├── mesh.npz                  vertices, faces — la reference
    ├── cameras.json              intrinseques + 18 poses camera->monde en 4x4
    ├── images/<vue>.png          RGB, fond noir
    └── masks/<vue>.png           silhouette binaire
```

`data/` n'est jamais commite. Le dataset se reconstruit en 5 s.

## L'assertion qui compte

`tests/integration/test_benchmark.py` re-rend le mesh stocke avec la pose et les intrinseques
stockes, et exige de retrouver **exactement** le masque stocke. Des poses et des masques qui
divergent empoisonneraient toutes les metriques de F12 sans jamais lever d'erreur.

## Metriques

Definies et hierarchisees dans `docs/methodology.md`. Rappel : IoU de silhouette et LPIPS en P1,
F-score@tau en P2, Chamfer en diagnostic uniquement. Les seuils vivront dans
`bench/thresholds.json` et **ne peuvent que monter**.
