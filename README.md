# PixaBoost

Reconstruction 3D (GLB) d'une piece photographiee, pour un usage **visualisation / catalogue**,
via inference [Pixal3D](https://github.com/TencentARC/Pixal3D) multi-vues.

## Demarrer

```bash
uv run poe setup     # installe l'env CPU depuis zero (Python 3.11 pinne)
uv run poe check     # lint + types + tests, < 60 s, sans GPU ni reseau
```

## GUI d'essais

```powershell
uv sync --extra gui
uv run pixaboost-gui
```

La fenetre lance les essais CPU verifies (`check`, `test`, `bench-build`) et une reconstruction
mono-vue sur un **Pod deja actif**. Elle affiche la commande sanitisee, la phase, la progression
disponible, les sorties, la duree, le code retour et les artefacts produits.

Le cache est verifie avant toute connexion : un hit ne demande aucune confirmation et n'ouvre
aucune session SSH. Un miss exige une autorisation ephemere explicite. La GUI ne provisionne ni
ne demarre de Pod et n'achete, ne recharge ou n'active jamais de credits.

## Ou lire quoi

| Fichier | Contenu |
|---|---|
| [`CLAUDE.md`](./CLAUDE.md) | **Point d'entree.** Commandes, contraintes dures, routine de session. |
| [`docs/methodology.md`](./docs/methodology.md) | Ce qu'on cherche a prouver et comment. Le gate F13. |
| [`PROGRESS.md`](./PROGRESS.md) | Etat verifie et prochaine action. |
| [`feature_list.json`](./feature_list.json) | Etat machine des features et leurs commandes de verification. |
| [`DECISIONS.md`](./DECISIONS.md) | Pourquoi les choix ont ete faits. |
| [`docs/gui.md`](./docs/gui.md) | Architecture et limites honnetes de la GUI d'essais. |

## Etat

Sprint 0 — harness, GUI d'essais (F08) et transport `ssh-pod` (F09) livres et verifies **sur CPU**.
La preuve materielle de bout en bout (F07) n'a pas encore tourne : voir `PROGRESS.md`.
