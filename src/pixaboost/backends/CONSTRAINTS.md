# `backends/` — adaptateurs

## Role

Traduire entre un artefact externe (modele GPU, service distant, fichier telecharge) et un type
de `core/`. **Rien d'autre.**

## Contraintes

1. **Aucune logique metier.** Si une fonction d'ici prend une decision d'algorithme, calcule une
   metrique ou choisit un seuil, elle est au mauvais endroit : elle appartient a `core/`.
   Test de discrimination : *ce code changerait-il si on remplacait Pixal3D par un autre modele ?*
   Si non, ce n'est pas un backend.
2. **Toute conversion de convention se fait ici, a la frontiere.** Les poses, les axes, les
   unites et les ordres d'axes sont normalises vers les conventions de `core/ARCHITECTURE.md`
   des l'entree. `core/` ne doit jamais rencontrer une convention etrangere.
3. **`vendor/pixal3d/` est en lecture seule**, pinne sur un SHA. On n'edite jamais le code amont
   en place : toute adaptation vit dans `pixal3d.py`. Sinon un `git pull` amont detruit le travail
   et le comportement devient irreproductible.
4. **Tout appel GPU passe par le cache** (`cache.py`) et respecte l'idempotence : un artefact
   deja present n'est jamais regenere. Le GPU est distant et facture a la seconde.
5. **Tout appel GPU ecrit son manifest** dans `runs/<ts>/manifest.json` : sha git, revision du
   modele, seeds, parametres, poses. Un resultat sans manifest n'est pas reproductible, donc il
   ne prouve rien.

## Modules

| Module | Responsabilite |
|---|---|
| `pixal3d.py` | Adaptateur Pixal3D, mono-vue et multi-vues |
| `pose.py` | Estimation de poses par VGGT sur des photos non posees |
| `perceptual.py` | LPIPS (torch CPU) — hors de `core/` par ADR-0003 |
| `cache.py` | Cache d'artefacts adresse par contenu |

## Pourquoi cette couche est mince

Sans GPU local, la boucle de developpement ne tient que si la metrologie s'execute sur des
artefacts en cache. Plus `backends/` est mince, plus la part du projet iterable en CPU local est
grande. Toute logique qui remonte ici est de la logique qu'on ne peut plus tester gratuitement.
