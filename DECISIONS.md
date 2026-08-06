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

## ADR-0004 — Python 3.11

**Date** : 2026-08-06 · **Statut** : accepte

**Decision.** Le projet est pinne sur Python 3.11 (`.python-version`, `requires-python`).

**Raison.** La machine de dev a Python 3.13, mais la stack scientifique requise en aval —
`open3d` pour l'ICP, l'ecosysteme `torch` cote GPU — accuse un retard de support sur les
versions recentes. Decouvrir une incompatibilite au moment de brancher le recalage couterait
une reinstallation complete de l'environnement. `uv` installe 3.11 automatiquement, donc le
pin ne coute rien.

**Reexamen.** Quand `open3d` et `torch` supporteront 3.13 en version stable.
