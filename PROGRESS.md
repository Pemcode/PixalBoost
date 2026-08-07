# PROGRESS

> Journal d'etat entre sessions. La session qui arrive lit ce fichier en premier ;
> la session qui part le met a jour avant de commiter. Voir la routine dans `CLAUDE.md`.

## Etat verifie

- **Sprint** : 0 (harness minimal + spike de lecture de code)
- **Features `passing`** : F00, F01, F02, F03, F04, F05
- **Feature active** : aucune. F06 et F07 sont **bloquees** (voir plus bas).
- **Dernier gate vert** : `uv run poe check` — ruff + mypy strict + **121 tests**, **3,6 s**

## Fait

**F00 — harness.** `uv` + Python 3.11 pinne, `uv.lock` commite. Surface `uv run poe
{setup,lint,fmt,typecheck,test,check,bench-build}` plus un `Makefile` delegateur (ADR-0002).
CI CPU GitHub Actions. `.gitattributes` pour forcer les fins de ligne en LF.

**F01 — spike Pixal3D.** Submodule `vendor/pixal3d` pinne sur `cdbb2bb`. Verdict dans
`docs/pixal3d-internals.md`, verrouille par 7 tests de contrat en analyse statique `ast`.

**F02 — `core/geometry.py`.** Sim3 (composition, inverse, matrice), camera Blender, projection
et back-projection exactement inversibles, grille de conditionnement canonique. 20 assertions
analytiques.

**F03 — `core/render.py` + `core/metrics.py`.** Rasteriseur numpy pur avec z-buffer et
interpolation de profondeur perspective-correcte ; IoU de silhouette, F-score, Chamfer,
echantillonnage de surface pondere par l'aire. 34 assertions analytiques.

**F04 — frontiere `core/`.** Detection statique : pas de torch/GPU/reseau, dependances dirigees
vers l'interieur, pas d'alea global implicite. Les detecteurs sont eux-memes testes, et le
garde-fou a ete valide en injectant une vraie violation dans `core/`.

**F05 — benchmark synthetique.** 3 pieces procedurales etanches (equerre, arbre etage, rondelle
percee) x 18 vues en 512, construites en 5,1 s hors ligne. Voir `docs/benchmark.md`.

## Ce que F01 a etabli — a lire avant F10

**Le multi-vues n'est pas expose, il est bloque** par un `assert transform_matrix is None`
(`image_conditioned_proj.py:211`). Le « 2 views by default » du README amont est de
l'echantillonnage a l'entrainement, pas du conditionnement multi-vues.

Deux consequences, actees en ADR-0005 :
1. **F10 devient une implementation**, pas un wrapper. L'intervention reste chirurgicale : la
   plomberie mathematique existe deja et accepte une camera arbitraire.
2. **H1 est plus fragile qu'anticipe.** Le checkpoint publie est entraine en mono-vue a camera
   fixe ; un volume de features moyenne sur N vues est hors distribution pour ces poids. F12
   devra mesurer une variante de repli en plus de B1.

## Bloque — demande une action de l'utilisateur

**F06 (image Docker GPU -> GHCR)** et **F07 (smoke RunPod)** ne peuvent pas etre verifiees
en l'etat. Leurs commandes de verification exigent des ressources externes absentes :

| Feature | Ce qui manque | Ce qu'il faut |
|---|---|---|
| F06 | Aucun remote git configure ; `docker` n'est pas installe en local | Creer le depot distant GitHub et le declarer en `origin`, pour que le workflow puisse builder et pousser sur GHCR |
| F07 | Aucune cle API RunPod, aucun endpoint | Un compte RunPod, une cle API dans `.env`, et un network volume pour les ~26 Go de poids |

Conformement a la contrainte n°2, elles restent `not_started` : le code peut etre ecrit, mais
tant que la commande de verification n'a pas tourne, la feature n'est pas faite.

## Prochaine action

Deux voies possibles, au choix de l'utilisateur :

1. **Debloquer F06/F07** en fournissant remote GitHub et cle RunPod, ce qui ferme le Sprint 0.
2. **Attaquer le Sprint 1 par sa partie CPU** : F11 (`core/registration.py`, estimation de Sim3
   par RANSAC + ICP et score de confiance de pose) est entierement testable hors GPU et sur le
   benchmark synthetique. C'est le prerequis de F10 de toute facon, puisque le multi-vues exige
   des poses relatives.

Recommandation : **la voie 2**, parce qu'elle avance le chemin critique sans dependre d'un
acces externe, et parce que le recalage est le maillon faible identifie des le plan.

## Jeu de photos reelles disponible

L'utilisateur dispose d'une serie de **18 photos** d'une **piece metallique mate** :
**6 azimuts x 3 elevations** (+45 deg, 0 deg, -45 deg). Elles seront ajoutees au depot le moment
venu (probablement en amont de F12).

Le benchmark synthetique de F05 reproduit deja **exactement** cette geometrie de prise de vue.
Piece **mate** : bonne nouvelle, les reflets speculaires sont le pire cas pour l'estimation de
pose comme pour le detourage.

## Notes pour la session suivante

- `docker` et `make` ne sont pas installes sur la machine de dev (Windows). Assume : l'image GPU
  se construit en CI, `poe` remplace `make` en local.
- Aucun GPU local. Tout ce qui touche au GPU passe par RunPod et doit remplir `artifacts/`.
- Cout Pixal3D releve en F01 : ~18 Go de VRAM en standard, ~10-12 Go en `--low_vram`.
