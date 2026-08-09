# Runtime GPU

> A lire avant de lancer quoi que ce soit sur RunPod ou de toucher a `docker/`.

## Pourquoi une image construite en CI

L'environnement Pixal3D est hostile a reproduire a la main. Le reconstruire en SSH a chaque
session etait le risque n°2 du projet, apres l'algorithme lui-meme. On le construit **une fois**,
on le pousse sur GHCR, et pods comme workers serverless demarrent tous des memes octets.

## Ce que le spike F06 a etabli

`vendor/pixal3d/requirements-hfdemo.txt` epingle des **wheels precompiles** pour *toutes* les
extensions CUDA : `natten`, `flash_attn_3`, `o_voxel`, `flex_gemm`, `cumesh`, `nvdiffrast`,
`nvdiffrec_render`. Consequences directes :

- **Rien n'est compile.** L'image de base est `runtime`, pas `devel` ; pas de `nvcc`, pas de
  `NATTEN_CUDA_ARCH`, pas de build de 45 minutes. Le README amont decrit une compilation depuis
  les sources : c'est le chemin difficile, et il est evitable.
- **Les wheels sont `cp310`**, donc l'interpreteur doit etre **Python 3.10 exactement**. C'est la
  raison du pin sur `ubuntu22.04`, dont le python systeme est 3.10.
- Le couple est **torch 2.6.0 / CUDA 12.4**. Changer l'un invalide tous les wheels.

Les 8 URLs de wheels ont ete verifiees joignables avant d'ecrire le Dockerfile.

## Les poids ne sont pas dans l'image

~26 Go : `Pixal3D` (~24 Go) + `DINOv3` + `MoGe-2` + `BiRefNet`. Les embarquer rendrait l'image
impossible a pousser. Ils vivent sur un **network volume**, telecharges dans `HF_HOME` au premier
demarrage.

**Consequence sur l'UX de production** : un cold start se compte en **minutes**, pas en 200 ms.
L'API de production devra etre **asynchrone** (soumission puis polling), jamais `/runsync`.

## Couts et VRAM

| Mode | VRAM | Resolution par defaut |
|---|---|---|
| Standard | ~18 Go | 1536 |
| `--low_vram` | ~10-12 Go | 1024 |

Le mode `low_vram` charge les modeles a la demande par etage. Il est plus lent mais tient sur des
GPU nettement moins chers. **Par defaut le handler utilise `low_vram=True`** : en phase recherche
on remplit un cache d'artefacts en batch, le debit compte plus que la latence unitaire.

## Phases

| Phase | Ressource | Pourquoi |
|---|---|---|
| Recherche | **Pod persistant** + network volume | On lance des batchs qui remplissent `artifacts/`, puis on eteint. Pas de cold start repete. |
| Production | **Serverless** + network volume | Scale-to-zero, mais cold start en minutes -> API asynchrone obligatoire. |

## Le cache d'artefacts est le socle du developpement

Sans GPU local, la boucle de dev ne tient que par `backends/cache.py`. Une fois un GLB produit,
toute la metrologie tourne en CPU, en local, gratuitement.

Deux invariants a ne jamais affaiblir :

1. **La cle couvre la revision du modele.** Bouger le submodule `vendor/pixal3d` invalide tous les
   artefacts. Sinon on comparerait les sorties de deux modeles differents en appelant l'ecart un
   resultat.
2. **`meta.json` est ecrit en dernier et fait seul foi.** Un GLB sans lui est un transfert
   interrompu, pas un hit de cache. Un fichier tronque annonce comme present empoisonnerait toutes
   les metriques en aval sans jamais lever d'erreur.

Et la contrainte n°9 de `CLAUDE.md` : **ne jamais regenerer un artefact deja present**. Le GPU est
distant et facture a la seconde. `ArtifactCache.store` refuse d'ecraser par defaut ; il faut
demander `overwrite=True` explicitement.

## Configurer un Pod RunPod (phase recherche)

**Template**

| Champ | Valeur |
|---|---|
| Container Image | `ghcr.io/pemcode/pixalboost:gpu-latest` |
| Container Start Command | `sleep infinity` |
| Container Disk | 30 Go |
| Volume Mount Path | `/workspace` |
| Network Volume | 60 Go, **meme region que le GPU** |
| Environment | `HF_HOME=/workspace/huggingface` |
| GPU | >= 16 Go de VRAM (RTX 4090, L40S, A100) |

**Les deux reglages qui cassent tout si on les oublie :**

1. **`Container Start Command: sleep infinity`.** La commande par defaut de l'image est
   `python3 handler.py`, qui demarre la boucle serverless RunPod. Dans un Pod elle n'a aucun job
   a consommer : le conteneur tourne dans le vide ou sort. On l'ecrase pour garder un shell.
2. **`HF_HOME=/workspace/huggingface`.** L'image pointe par defaut sur `/runpod-volume`, qui est
   le chemin de montage **serverless**. Sur un Pod le volume est monte sur `/workspace`. Sans ce
   reglage, les ~26 Go de poids atterrissent sur le disque du conteneur : re-telecharges a chaque
   redemarrage, et le disque peut saturer.

**Verifier, dans cet ordre**

```bash
# 1. L'environnement est complet (instantane, aucun poids telecharge)
python3 /opt/pixaboost/verify_extensions.py

# 2. Le GPU est vu
python3 -c "import torch; print(torch.cuda.get_device_name(0))"

# 3. Premiere inference: telecharge ~26 Go dans HF_HOME, puis genere
python3 /opt/pixal3d/inference.py \
    --image /opt/pixal3d/assets/images/0_img.png \
    --output /workspace/test.glb --low_vram

# 4. Les poids sont bien sur le volume, pas sur le disque conteneur
du -sh /workspace/huggingface
```

L'etape 1 est celle qui merite d'etre lancee en premier : elle echoue en une seconde si l'image
est cassee, au lieu d'attendre 26 Go de telechargement pour l'apprendre.

## Secrets

La cle API RunPod se lit depuis `RUNPOD_API_KEY` ou depuis `runpod.env` (jeton nu sur une ligne,
ou `NOM=valeur`). Ce fichier **n'est jamais commite** : `.gitignore` couvre `*.env`, `*.key` et
`*secret*`.

`.env` seul ne suffisait pas — il ne matche pas `runpod.env`. C'est le genre de motif trop etroit
qui laisse fuiter une cle au premier `git add -A`.

Le client ne place jamais la cle dans un `repr`, un log ou un message d'exception. Un test le
verifie.

## Publication de l'image

`.github/workflows/gpu-image.yml`, sur `workflow_dispatch` ou sur push touchant `docker/**` ou le
pointeur de submodule.

- Le runner libere d'abord ~20 Go : les ~14 Go disponibles par defaut ne suffisent pas.
- Le nom d'image est **mis en minuscules a l'execution** : GHCR refuse toute majuscule, et le
  login du proprietaire est en casse mixte. C'est ce qui a fait echouer le premier build.
- Le SHA amont epingle est estampille dans l'image (`PIXAL3D_SHA`) et renvoye dans chaque
  resultat de job, pour qu'un artefact en cache soit toujours tracable jusqu'a la revision exacte
  qui l'a produit.
- Une verification d'imports en fin de build echoue si une extension manque — mieux vaut casser
  un build CPU gratuit qu'une heure de GPU facturee.

Tags publies : `:gpu-latest` et `:gpu-<sha>`.
