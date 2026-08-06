# PixaBoost

Reconstruction 3D (GLB) d'une piece photographiee, pour un usage **visualisation / catalogue**,
via inference [Pixal3D](https://github.com/TencentARC/Pixal3D) multi-vues.

## Demarrer

```bash
uv run poe setup     # installe l'env CPU depuis zero (Python 3.11 pinne)
uv run poe check     # lint + types + tests, < 60 s, sans GPU ni reseau
```

## Ou lire quoi

| Fichier | Contenu |
|---|---|
| [`CLAUDE.md`](./CLAUDE.md) | **Point d'entree.** Commandes, contraintes dures, routine de session. |
| [`docs/methodology.md`](./docs/methodology.md) | Ce qu'on cherche a prouver et comment. Le gate F13. |
| [`PROGRESS.md`](./PROGRESS.md) | Etat verifie et prochaine action. |
| [`feature_list.json`](./feature_list.json) | Etat machine des features et leurs commandes de verification. |
| [`DECISIONS.md`](./DECISIONS.md) | Pourquoi les choix ont ete faits. |

## Etat

Sprint 0 — bootstrap du harness. Voir `PROGRESS.md`.
