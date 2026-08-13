# PixaBoost

Reconstruction 3D (GLB) d'une piece photographiee, pour un usage **visualisation / catalogue**.
Approche : inference [Pixal3D](https://github.com/TencentARC/Pixal3D) **multi-vues** sur N photos.

**Ou on en est** : lis `PROGRESS.md` puis `feature_list.json`. Ce sont les deux seules sources
de verite sur l'etat du projet. Ne deduis jamais l'etat depuis le code.

---

## Demarrer

```bash
uv run poe setup     # installe l'env CPU depuis zero
uv run poe check     # lint + types + tests. DOIT etre vert avant tout commit.
```

`make <task>` fait la meme chose sur Linux/CI/RunPod. Sur Windows, utilise `uv run poe`.

## Verifier

| Commande | Ce qu'elle prouve |
|---|---|
| `uv run poe check` | Gate CPU complet. < 60 s, sans GPU ni reseau. |
| `uv run poe test` | Tests unitaires + integration seulement. |
| `uv run pytest tests/e2e -m gpu` | Chaine reelle sur GPU. Jamais en CI. |

Chaque feature de `feature_list.json` porte sa propre commande de verification. **C'est elle
qui fait foi**, pas ton jugement.

---

## Contraintes dures (non negociables)

1. **WIP = 1.** Une seule feature `active` a la fois. Interdiction de « refactorer au passage » :
   c'est le premier destructeur de jugement d'achevement.
2. Une feature passe a `passing` **uniquement** quand sa commande de verification s'execute avec
   succes. Tu ne declares jamais toi-meme qu'une tache est terminee.
3. **TDD strict dans `src/pixaboost/core/`** : test ecrit, execute, et **constate rouge** avant
   la moindre ligne d'implementation.
4. `core/` ne doit **jamais** importer `torch`, faire du reseau, ni toucher au GPU.
   Verifie par `tests/unit/test_architecture.py`.
5. `backends/` ne contient **aucune logique metier** : uniquement de la traduction entre un
   artefact externe et un type de `core/`.
6. `vendor/pixal3d/` est en **lecture seule**, pinne sur un SHA. Toute adaptation vit dans
   `backends/pixal3d.py`.
7. Toute experience ecrit `runs/<ts>/manifest.json` (sha git, revision modele, seeds, params,
   poses). **Sans manifest, un resultat est nul et non avenu.**
8. Les seuils de `bench/thresholds.json` ne peuvent que **monter**, jamais descendre.
9. Ne regenere jamais un artefact GPU deja present dans `artifacts/` (le GPU est distant et payant).
10. Toute decision d'architecture ou d'algorithme -> une entree dans `DECISIONS.md`.
11. **Aucun code de fusion ou de confiance tant que le gate F13 n'a pas statue.** Voir
    `docs/methodology.md`. C'est la contrainte anti-overreach centrale du projet.
12. Langue : documents `.md` en francais, code et docstrings en anglais. Ne melange pas.

---

## Carte du depot

```
src/pixaboost/core/       CPU pur, deterministe, 100 % TDD    -> lis core/ARCHITECTURE.md
src/pixaboost/backends/   Adaptateurs GPU / reseau, fins       -> lis backends/CONSTRAINTS.md
src/pixaboost/bench/      Rendu synthetique + runner d'eval
src/pixaboost/trials/     Orchestration d'un essai + production du manifeste -> ADR-0012
src/pixaboost/gui/        Harness PyQt6 d'essais CPU observables
tests/unit/               CPU, < 60 s, sans reseau ni GPU
tests/e2e/                GPU reel, marques @pytest.mark.gpu
artifacts/                Cache d'artefacts GPU (jamais commite)
runs/<ts>/                manifest.json, metrics.json, logs.jsonl, PNG comparatifs
vendor/pixal3d/           Submodule Pixal3D, lecture seule
```

## Docs thematiques

| Document | A lire quand |
|---|---|
| `docs/methodology.md` | **Toujours en premier.** Hypotheses, protocole, et le gate F13. |
| `docs/testing.md` | Avant d'ecrire le moindre test. Regles TDD et decoupage CPU/GPU. |
| `docs/pixal3d-internals.md` | Avant de toucher a `backends/pixal3d.py`. |
| `docs/benchmark.md` | Avant de toucher a `bench/` ou aux metriques. |
| `docs/gpu-runtime.md` | Avant de lancer quoi que ce soit sur RunPod ou de toucher a `docker/`. |
| `docs/gui.md` | Architecture, commandes et exclusions de la GUI d'essais. |
| `docs/segmentation.md` | Avant de toucher a la segmentation par clic (SAM 3, `core/segmentation.py`). |
| `DECISIONS.md` | Avant de remettre en cause un choix : la raison y est deja. |

---

## Routine de session

**Arrivee** — dans cet ordre, sans exception :
1. Lire `PROGRESS.md` (etat verifie, blocages, prochaine action).
2. Lire `feature_list.json` (quelle feature est `active`).
3. Lancer `uv run poe check` pour confirmer que le depot est sain **avant** de le modifier.

**Sortie** — une session n'est pas finie tant que les cinq points ne sont pas vrais :
- [ ] `uv run poe check` est vert
- [ ] `feature_list.json` est a jour (etat + evidence)
- [ ] `PROGRESS.md` est a jour (fait / en cours / bloque / prochaine action)
- [ ] Aucun residu de debug (`print`, `breakpoint`, code commente, TODO orphelin)
- [ ] Travail commite ; la session suivante peut demarrer sans intervention manuelle

**Le point le plus important de ce fichier** : ton estimation de l'achevement d'une tache n'a
aucune valeur. Seule la commande de verification en a. Si elle n'a pas tourne, la feature
n'est pas faite.
