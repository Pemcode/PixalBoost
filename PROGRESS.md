# PROGRESS

> Journal d'etat entre sessions. La session qui arrive lit ce fichier en premier ;
> la session qui part le met a jour avant de commiter. Voir la routine dans `CLAUDE.md`.

## Etat verifie

- **Sprint** : 0 (harness minimal + spike de lecture de code)
- **Features `passing`** : F00, F01
- **Feature active** : F02 — `core/geometry.py`
- **Dernier gate vert** : `uv run poe check` — ruff + mypy strict + **9 tests**, **1,15 s**

## Fait

**F00 — harness.** Depot git sur `main`. Environnement CPU reproductible : `uv` + Python 3.11
pinne, `uv.lock` commite. Surface de commandes `uv run poe {setup,lint,fmt,typecheck,test,check}`
plus un `Makefile` delegateur (ADR-0002). Squelette `core/` + `backends/`, CLI, CI CPU GitHub
Actions, `.gitattributes` pour forcer les fins de ligne en LF.

**F01 — spike Pixal3D.** Submodule `vendor/pixal3d` pinne sur `cdbb2bb`. Verdict ecrit dans
`docs/pixal3d-internals.md`, verrouille par 7 tests de contrat en analyse statique `ast`
(donc sans torch, et qui servent de detecteur de derive si on bouge le SHA).

## Ce que F01 a etabli — a lire avant F10

**Le multi-vues n'est pas expose, il est bloque** par un `assert transform_matrix is None`
(`image_conditioned_proj.py:211`). Le « 2 views by default » du README amont est de
l'echantillonnage a l'entrainement, pas du conditionnement multi-vues.

Deux consequences, actees en ADR-0005 :
1. **F10 devient une implementation**, pas un wrapper. Mais l'intervention reste chirurgicale :
   la plomberie mathematique existe deja et accepte une camera arbitraire.
2. **H1 est plus fragile qu'anticipe.** Le checkpoint publie est entraine en mono-vue a camera
   fixe ; un volume de features moyenne sur N vues est hors distribution pour ces poids. F12
   devra mesurer une variante de repli en plus de B1.

## En cours

F02 — `core/geometry.py`. Rien d'ecrit encore.

## Bloque

Rien.

## Prochaine action

**F02, en TDD strict.** Ecrire d'abord `tests/unit/test_geometry.py` et le constater rouge.
Les conventions a implementer sont relevees dans `docs/pixal3d-internals.md` (section « Modele
de camera ») : convention Blender, camera selon `-Z`, capteur 32 mm, focale `16/tan(fov/2)`,
grille dans `[-1,1]` tournee par `[[1,0,0],[0,0,-1],[0,1,0]]`.

Assertions analytiques attendues : un Sim3 compose de son inverse donne l'identite ; projeter
puis back-projeter un point a sa profondeur connue redonne le point de depart.

## Jeu de photos reelles disponible

L'utilisateur dispose d'une serie de **18 photos** d'une **piece metallique mate** :
**6 azimuts x 3 elevations** (+45 deg, 0 deg, -45 deg). Elles seront ajoutees au depot le moment
venu (probablement en amont de F12).

**Consequence actee sur F05** : le benchmark synthetique doit rendre **exactement cette geometrie
de prise de vue** (6 x 3 = 18 vues). Sans cela, la comparaison synthetique -> reel melange l'effet
du domaine et l'effet du protocole de capture, et ne mesure plus rien d'interpretable.

Piece **mate** : c'est une bonne nouvelle. Les reflets speculaires sont le pire cas pour
l'estimation de pose comme pour le detourage ; une surface mate les evite en grande partie.

## Notes pour la session suivante

- `docker` et `make` ne sont pas installes sur la machine de dev (Windows). Assume : l'image GPU
  se construit en CI, `poe` remplace `make` en local.
- Aucun GPU local. Tout ce qui touche au GPU passe par RunPod et doit remplir `artifacts/`.
- Cout Pixal3D releve en F01 : ~18 Go de VRAM en standard, ~10-12 Go en `--low_vram`.
  A prendre en compte pour le choix de GPU RunPod en F06/F07.
