# DECISIONS

> Journal des decisions d'architecture et d'algorithme. Format : decision, raison,
> alternatives rejetees, condition de reexamen. **Avant de remettre en cause un choix,
> lis l'entree correspondante : la raison y est deja.**

---

## ADR-0001 — Mesurer le multi-vues natif de Pixal3D avant de construire une fusion maison

**Date** : 2026-08-06 · **Statut** : accepte

**Decision.** Le premier livrable est l'implementation et l'**evaluation chiffree** du chemin
multi-vues natif de Pixal3D. La fusion late ponderee par des scores de confiance par voxel —
l'idee initiale du projet — est repoussee en Sprint 2 **conditionnel**, debloquee uniquement par
un diagnostic d'erreur qui la justifie (gate F13).

**Raison.** La recherche a etabli que Pixal3D fait deja de la fusion multi-vues, en *early
fusion* : les features DINOv3 back-projetees de chaque vue sont moyennees par voxel avant la
generation. Le vrai baseline a battre n'est donc pas le mono-vue mais le multi-vues natif.
Construire la machinerie de fusion avant de l'avoir mesure serait un investissement sur une
hypothese non testee, pour un gain potentiellement nul. L'usage vise (visualisation / catalogue)
place la barre sur la silhouette et l'apparence, precisement ce que le multi-vues natif corrige
en premier — ce qui rend l'issue « B1 suffit » probable.

**Alternatives rejetees.**
- *Construire la fusion ponderee directement* : rejete, c'est l'overreach que le harness doit
  empecher. Cout eleve, gain non demontre.
- *Ne jamais explorer la fusion* : rejete. La late fusion reste le seul moyen d'appliquer une
  contrainte geometrique dure (enveloppe visuelle) capable d'opposer un veto a l'hallucination,
  ce que l'early fusion ne peut structurellement pas faire. On la garde en reserve.

**Reexamen.** Au gate F13, sur la base de `runs/<ts>/metrics.json`.

---

## ADR-0002 — `poethepoet` comme runner canonique, `make` en delegateur

**Date** : 2026-08-06 · **Statut** : accepte

**Decision.** La surface de commandes canonique est `uv run poe <task>`, declaree dans
`pyproject.toml`. Un `Makefile` de six lignes delegue vers `poe`, pour que la convention
`make check` reste valable la ou GNU make existe (CI Linux, pods RunPod).

**Raison.** `make` n'est pas installe sur la machine de developpement (Windows + Git Bash) et
l'imposer ajouterait une dependance systeme a installer manuellement — exactement le genre de
friction d'environnement que le harness doit supprimer. `poethepoet` est installe par
`uv sync`, donc l'environnement reste auto-descriptif et reproductible.

**Alternatives rejetees.**
- *Installer GNU make sur Windows* : friction systeme, et rend `make setup` non auto-portant.
- *Scripts `.ps1` + `.sh` en double* : deux surfaces a maintenir, elles divergeront.
- *`just`* : meme probleme d'installation systeme que `make`.

**Reexamen.** Si le projet cesse d'etre developpe depuis Windows.

---

## ADR-0003 — LPIPS vit hors de `core/`

**Date** : 2026-08-06 · **Statut** : accepte

**Decision.** Les metriques de geometrie pure (IoU de silhouette, F-score, Chamfer) vivent dans
`core/metrics.py`. La metrique perceptuelle LPIPS vit dans `backends/perceptual.py`.

**Raison.** LPIPS est un reseau de neurones : il exige `torch` et telecharge ses poids au premier
appel. Cela viole deux contraintes dures de `core/` (pas de torch, pas de reseau). Or ces
contraintes sont ce qui garantit un gate CPU de moins de 60 secondes, executable hors ligne et
en CI gratuite. On preserve la contrainte plutot que de l'affaiblir pour une seule metrique.

**Consequence.** LPIPS n'entre pas dans `poe check`. Il est calcule dans le runner d'evaluation
(`bench-compare`), dont les resultats sont archives dans `runs/<ts>/metrics.json`.

**Alternatives rejetees.**
- *Autoriser torch dans `core/`* : rejete, cela dissout la frontiere qui rend le gate rapide.
- *Renoncer a LPIPS* : rejete, c'est une metrique P1 pour l'usage catalogue.

---

## ADR-0005 — Le multi-vues doit etre implemente, et H1 est plus fragile qu'anticipe

**Date** : 2026-08-06 · **Statut** : accepte · **Origine** : spike F01

**Constat.** Le multi-vues de Pixal3D n'est pas expose : un `assert transform_matrix is None`
(`image_conditioned_proj.py:211`) force toute back-projection sur une camera front-view fixe. Le
« 2 views by default » du README amont designe un **echantillonnage a l'entrainement** — un
`view_idx` tire au hasard parmi deux rendus — pas un conditionnement multi-vues. Details et
preuves dans `docs/pixal3d-internals.md`.

**Decision 1.** F10 implemente le chemin multi-vues par sous-classement dans
`backends/pixal3d.py` : lever l'`assert`, recuperer le `valid_mask` jete en amont, et faire une
**moyenne masquee** par voxel sur N vues. `vendor/` reste intouche (contrainte n°6).
L'intervention est chirurgicale : `project_points_to_image_batch` accepte deja une matrice de
camera arbitraire, et le parametre est cable de bout en bout.

**Decision 2.** Le plan est amende : le checkpoint publie a ete entraine en conditionnement
**mono-vue a camera fixe**. Un volume de features moyenne sur plusieurs vues est donc **hors
distribution** pour ces poids. Rien dans le code publie ne prouve que *ces poids-la* ont vu ce
cas ; les resultats multi-vues du papier viennent peut-etre d'une variante non publiee.

**Consequence.** L'hypothese H1 (« B1 bat B0 ») est nettement plus fragile qu'estime au moment du
plan. F12 doit donc mesurer B1 **et** une variante de repli — une seule vue conditionne, les
autres ne servant qu'au recalage et a la texture — avant de conclure quoi que ce soit au gate F13.

**Ce que cela ne change pas.** L'ordre reste le bon : mesurer avant de construire. Si le
multi-vues natif s'avere inexploitable sur ces poids, c'est precisement le genre de diagnostic
qui debloque le Sprint 2 — et on l'aura obtenu pour le cout d'une lecture de code, pas d'un
pipeline de fusion complet.

**Reexamen.** Au gate F13.

---

## ADR-0004 — Python 3.11

**Date** : 2026-08-06 · **Statut** : accepte

**Decision.** Le projet est pinne sur Python 3.11 (`.python-version`, `requires-python`).

**Raison.** La machine de dev a Python 3.13, mais la stack scientifique requise en aval —
`open3d` pour l'ICP, l'ecosysteme `torch` cote GPU — accuse un retard de support sur les
versions recentes. Decouvrir une incompatibilite au moment de brancher le recalage couterait
une reinstallation complete de l'environnement. `uv` installe 3.11 automatiquement, donc le
pin ne coute rien.

**Reexamen.** Quand `open3d` et `torch` supporteront 3.13 en version stable.
