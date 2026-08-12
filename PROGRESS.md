# PROGRESS

> Journal d'etat entre sessions. La session qui arrive lit ce fichier en premier ;
> la session qui part le met a jour avant de commiter. Voir la routine dans `CLAUDE.md`.

## Etat verifie

- **Sprint 0 : termine.** F00 a F06 sont `passing`.
- **Sprint 1 : entame.** F11 est `passing`. F07 est `blocked`.
- **Dernier gate vert** : `uv run poe check` — ruff + mypy strict + **186 tests**, **~5 s**.
  Vert aussi sur GitHub Actions.

## Fait

| Feature | Etat | Preuve |
|---|---|---|
| F00 harness | `passing` | `poe setup && poe check` |
| F01 spike Pixal3D | `passing` | 7 tests de contrat `ast` |
| F02 `core/geometry` | `passing` | 20 assertions analytiques |
| F03 rendu + metriques | `passing` | 34 assertions analytiques |
| F04 frontiere `core/` | `passing` | 24 tests, garde-fou valide par injection |
| F05 benchmark | `passing` | 3 pieces x 18 vues en 5,1 s |
| F06 image GPU | `passing` | manifeste GHCR verifie |
| F11 recalage | `passing` | 25 tests, ADR-0006 et ADR-0007 |

## Le jalon de la session du 2026-08-12

**Premiere reconstruction de bout en bout reussie.** Un GLB de 37 Mo produit sur RTX 4090 depuis
l'image publiee. Artefact et manifeste dans `artifacts/smoke-20260812/`.

Il a fallu quatre correctifs pour y arriver, tous partageant la meme signature : **un
`pip install` vert ne prouve rien sur l'execution.** Aucun n'etait visible avant du materiel
facture.

1. GHCR refuse les majuscules du login proprietaire (build casse).
2. `ATTN_BACKEND=flash_attn` par defaut importe un module que le wheel ne fournit pas -> `sdpa`.
3. Triton compile un helper C **a l'execution** -> `gcc`/`g++` dans l'image.
4. Le wheel natten amont ne porte que `sm_90` -> build officiel SHI-Labs multi-architectures.

Chaque incident est devenu une assertion executable dans `verify_extensions.py`, qui casse
desormais un build CPU gratuit plutot qu'une session GPU.

## Bloque

**F07** — le test e2e vise un **endpoint serverless** alors que la phase recherche tourne sur un
**Pod**. Il manque un transport SSH dans `backends/`, derriere la meme interface que le client
serverless. Le cache, l'adaptateur et les 40 tests CPU ne bougeraient pas.

**Le push est rejete** : `! [remote rejected] main -> main (pre-receive hook declined)`.
**1 commit en attente** (le correctif natten). Tant que ce n'est pas resolu, l'image publiee
n'a pas le correctif et chaque nouveau pod exige `bash /workspace/setup_pod.sh`.
Le message complet du hook n'a pas encore ete capture — c'est peut-etre la protection anti-secrets
de GitHub. **A diagnostiquer en priorite.**

## Deux correctifs encore manuels dans le pod

Ils vivent sur le volume reseau et survivent aux redemarrages :

- `/workspace/setup_pod.sh` — reinstalle le natten officiel (ADR-0009)
- `/workspace/run_mit.py` — force BiRefNet MIT au lieu de RMBG-2.0 gated (ADR-0010)

Le premier disparaitra quand le push passera. Le second doit etre porte proprement dans
`backends/`.

## Ce qu'il faut savoir avant F10 et F12

1. **Le multi-vues de Pixal3D est bloque**, pas absent : `assert transform_matrix is None`
   (`image_conditioned_proj.py:211`). F10 est une implementation. ADR-0005.
2. **Le checkpoint est entraine en mono-vue a camera fixe.** Un volume de features moyenne sur
   N vues lui est hors distribution. H1 est fragile ; F12 doit mesurer une variante de repli.
3. **L'ICP raffine, il ne cherche pas** : bassin ~20 degres. Partir des poses nominales de la
   prise de vue, pas d'un recalage global. ADR-0006.
4. **Une inference coute ~25 min**, domine par les chargements de modeles et non par le calcul.
   3 pieces x 18 vues = ~22 h de GPU. **Il faut batcher avant de lancer F12.**
5. **GPU : compute capability 5.0 a 9.0 uniquement.** Blackwell (sm_120) est exclu. ADR-0008.

## Prochaine action

**Regarder `artifacts/smoke-20260812/test.glb`.** C'est la premiere sortie reelle du modele et
personne ne l'a encore inspectee. Question a trancher : la qualite est-elle dans le bon ordre de
grandeur pour un usage catalogue ? La reponse oriente tout le reste — si le mono-vue est deja
suffisant sur des objets simples, le gate F13 se rapproche.

Ensuite, dans l'ordre de valeur :
1. Debloquer le push (le correctif natten doit entrer dans l'image).
2. Porter le patch BiRefNet dans `backends/` plutot que dans un script sur le volume.
3. Batcher l'inference — prerequis economique de F12.
4. F10, le chemin multi-vues.

## Jeu de photos reelles disponible

18 photos d'une **piece metallique mate**, 6 azimuts x 3 elevations (+45/0/-45). Le benchmark
synthetique reproduit deja exactement cette geometrie. A ajouter au depot en amont de F12.

## Notes

- `docker` et `make` ne sont pas installes en local (Windows). Assume.
- Aucun GPU local : tout passe par RunPod.
- Volume reseau RunPod en region `euro`, ~26 Go de poids en cache, facture meme pod eteint
  (~4 EUR/mois). Ne pas le supprimer : il evite de retelecharger les poids.
- Cle SSH locale : `~/.ssh/depthscan_sp005_ed25519` (pas de `id_ed25519`).
- Les logs GitHub Actions exigent une authentification meme sur un depot public ; le workflow
  publie la queue d'un build echoue en annotation de check-run, lisible anonymement.
