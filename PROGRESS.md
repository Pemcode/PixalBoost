# PROGRESS

> Journal d'etat entre sessions. La session qui arrive lit ce fichier en premier ;
> la session qui part le met a jour avant de commiter. Voir la routine dans `CLAUDE.md`.

## Etat verifie

- **Sprint** : 0 (harness minimal + spike de lecture de code)
- **Feature active** : F00 — squelette du depot et gate CPU
- **Dernier gate vert** : `uv run poe check` — lint + mypy strict + 2 tests, **1,1 s**
- **Commit** : _(aucun commit encore)_

## Fait

- Depot git initialise sur `main`.
- Environnement CPU reproductible : `uv` + Python 3.11 pinne, `pyproject.toml`, verrou `uv.lock`.
- Surface de commandes stable : `uv run poe {setup,lint,fmt,typecheck,test,check}`,
  plus un `Makefile` delegateur pour Linux/CI/RunPod (ADR-0002).
- Squelette de paquet : `core/`, `backends/`, CLI, plus un test exemple qui s'execute reellement.
- Artefacts de harness : `CLAUDE.md`, `AGENTS.md`, `feature_list.json`, `DECISIONS.md`,
  `docs/methodology.md`, `docs/testing.md`.

## En cours

Rien. F00 est en attente de sa verification finale et de son commit.

## Bloque

Rien.

## Prochaine action

**F01 — spike de lecture de code sur Pixal3D.** Cloner le depot en `vendor/pixal3d/` (pinne sur
un SHA), localiser le chemin de back-projection, et repondre dans `docs/pixal3d-internals.md` a :
le multi-vues est-il expose a l'utilisateur, sous quelle signature, avec quelle convention
d'intrinseques et d'extrinseques ?

Cette question ne demande **ni GPU ni poids de modele** — uniquement de la lecture. Elle
conditionne l'ampleur de F10, donc elle passe avant tout le reste.

## Notes pour la session suivante

- `docker` et `make` ne sont pas installes sur la machine de dev (Windows). C'est assume :
  l'image GPU se construit en CI, et `poe` remplace `make` en local.
- Aucun GPU local. Tout ce qui touche au GPU passe par RunPod et doit remplir `artifacts/`.
