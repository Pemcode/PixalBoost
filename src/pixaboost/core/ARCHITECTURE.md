# `core/` — couche CPU pure

## Role

Toute la geometrie et toute la metrologie du projet. C'est la couche qui **decide** ; `backends/`
ne fait que lui fournir de la matiere.

## Contraintes (verifiees par `tests/unit/test_architecture.py`)

1. **Aucun import de `torch`**, ni de quoi que ce soit qui en depende.
2. **Aucun acces reseau** : pas de `requests`, `urllib`, `socket`, `huggingface_hub`.
3. **Aucun acces GPU.**
4. **Aucun I/O implicite** : une fonction de `core/` prend des tableaux et des dataclasses, et
   rend des tableaux et des dataclasses. La lecture de fichiers est concentree dans des fonctions
   `load_*` explicites et clairement identifiables.
5. **Deterministe** : meme entree, meme sortie, bit a bit. Tout aleatoire prend un `seed` explicite
   en parametre — jamais de `np.random` global.

Ces contraintes ne sont pas de la purete pour la purete : elles sont ce qui rend `poe check`
executable en moins de 60 secondes, hors ligne, et en CI gratuite. Voir `docs/testing.md`.

## Conventions geometriques

Fixees une fois ici, pour que tout le depot s'y tienne.

- **Poses** : `Sim3` (rotation, translation, **echelle**), jamais `SE3`. L'echelle est necessaire
  parce que Pixal3D produit des sorties pixel-alignees a l'echelle de la depth MoGe-2, qui differe
  d'une vue a l'autre. C'est aussi le point d'extension vers la fidelite dimensionnelle
  (`docs/methodology.md`).
- **Convention camera** : OpenCV — `x` vers la droite, `y` vers le bas, `z` vers l'avant
  (dans l'axe de visee). Toute pose importee d'une source externe est convertie a l'entree, dans
  `backends/`, jamais au milieu de `core/`.
- **Extrinseques** : matrice `world_to_camera`. Le nom des variables doit le dire
  (`T_cam_world`), jamais un `pose` ambigu.
- **Unites** : sans dimension jusqu'au Sprint 4. Les metriques geometriques sont normalisees par
  la diagonale de la bbox de la verite terrain.

## Modules

| Module | Responsabilite |
|---|---|
| `geometry.py` | `Sim3`, intrinseques, projection, back-projection |
| `metrics.py` | IoU de silhouette, F-score@tau, Chamfer — geometrie pure uniquement |
| `render.py` | Rendu offscreen de vues held-out (silhouettes et couleur) |
| `registration.py` | Estimation de Sim3, RANSAC + ICP, score de confiance de pose |

`fusion.py`, `carving.py` et `confidence/` **n'existent pas** et ne doivent pas etre crees tant
que le gate F13 n'a pas statue (contrainte dure n°11 de `CLAUDE.md`).
