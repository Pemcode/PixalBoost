# PROGRESS

> Journal d'etat entre sessions. La session qui arrive lit ce fichier en premier ;
> la session qui part le met a jour avant de commiter. Voir la routine dans `CLAUDE.md`.

## Etat verifie

- **Sprint 0 : termine.** F00 a F06 sont `passing`.
- **Sprint 1 : commence.** F11 est `passing`.
- **Bloque** : F07 (endpoint RunPod, voir plus bas).
- **Dernier gate vert** : `uv run poe check` — ruff + mypy strict + **186 tests**, **4,9 s**.
  Vert aussi sur GitHub Actions, donc sur une machine Linux propre.

## Fait

**F00 — harness.** `uv` + Python 3.11 pinne. Surface `uv run poe
{setup,lint,fmt,typecheck,test,check,bench-build}` + `Makefile` delegateur (ADR-0002).

**F01 — spike Pixal3D.** Submodule pinne sur `cdbb2bb`. Verdict verrouille par 7 tests de
contrat en analyse statique `ast`.

**F02 — `core/geometry.py`.** Sim3, camera Blender, projection/back-projection inversibles.

**F03 — `core/render.py` + `core/metrics.py`.** Rasteriseur numpy pur, IoU de silhouette,
F-score, Chamfer, echantillonnage pondere par l'aire.

**F04 — frontiere `core/`.** Detection statique, detecteurs eux-memes testes, garde-fou valide
par injection d'une vraie violation.

**F05 — benchmark synthetique.** 3 pieces etanches x 18 vues (6 azimuts x 3 elevations) en 5,1 s.

**F06 — image GPU.** `ghcr.io/pemcode/pixalboost:gpu-latest` publiee, 6,24 Go compresses,
manifeste verifie recuperable anonymement. Voir `docs/gpu-runtime.md`.

**F11 — `core/registration.py`.** Umeyama closed-form, ICP tronque avec pre-alignement grossier,
score de confiance et refus. Deux mesures actees en ADR-0006 et ADR-0007.

## Les trois constats a connaitre avant F10

1. **Le multi-vues de Pixal3D est bloque**, pas seulement absent : `assert transform_matrix is
   None` (`image_conditioned_proj.py:211`). F10 est donc une implementation. ADR-0005.
2. **Le checkpoint publie est entraine en mono-vue a camera fixe.** Un volume de features moyenne
   sur N vues lui est hors distribution. H1 est fragile ; F12 doit mesurer une variante de repli.
3. **L'ICP raffine, il ne cherche pas** : bassin mesure a ~20 degres. Il ne recuperera pas une
   pose depuis rien. ADR-0006 laisse deux voies pour F10 — partir des **poses nominales de la
   prise de vue** (chemin privilegie, elles sont connues a quelques degres pres), ou ajouter un
   recalage global par descripteurs. **On n'ecrit pas le second avant d'avoir essaye le premier.**

## Bloque — demande une action de l'utilisateur

**F07 — smoke RunPod.** Le code est ecrit et couvert par 40 tests CPU (cache, client,
adaptateur) ; seule la verification reelle manque, donc la feature n'est pas faite.

Il manque deux choses :
- **Une cle API RunPod valide.** L'ancienne a ete exposee dans une sortie de terminal et doit
  etre revoquee et regeneree. Le code lit `runpod.env` a l'execution, donc la rotation est une
  simple edition de fichier.
- **Un endpoint deploye** a partir de `ghcr.io/pemcode/pixalboost:gpu-latest`, avec un network
  volume pour les ~26 Go de poids. Puis relancer avec `RUNPOD_ENDPOINT_ID=<id>`.

Le premier demarrage telecharge 26 Go et chaque cold start streame ces poids : plusieurs minutes
de GPU facture avant la moindre inference. C'est pour cela que rien n'a ete provisionne sans
accord explicite.

## Prochaine action

**F12 partiel, sans GPU** : cabler le runner d'evaluation (`bench/`) sur le benchmark
synthetique et sur `core/metrics.py`, avec des sorties de modele simulees. Cela met en place
`runs/<ts>/manifest.json`, `metrics.json` et la planche de rendus comparatifs, pour que le jour
ou les artefacts GPU arrivent, il ne reste qu'a les brancher.

Sinon, F10 des que F07 est debloquee.

## Jeu de photos reelles disponible

18 photos d'une **piece metallique mate**, 6 azimuts x 3 elevations (+45/0/-45). Le benchmark
synthetique reproduit deja exactement cette geometrie. A ajouter au depot en amont de F12.

## Notes

- `docker` et `make` ne sont pas installes en local (Windows). Assume.
- Aucun GPU local : tout passe par RunPod et doit remplir `artifacts/`.
- Cout Pixal3D : ~18 Go de VRAM en standard, ~10-12 Go en `--low_vram`.
- Les logs GitHub Actions exigent une authentification meme sur un depot public. Le workflow
  publie donc la queue d'un build echoue en **annotation de check-run**, lisible anonymement.
