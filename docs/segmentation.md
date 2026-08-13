# Segmentation par clic

> Contrat F14. A lire avant de toucher a `backends/sam3.py`,
> `core/segmentation.py` ou `gui/segmentation_view.py`.

## Le probleme que BiRefNet ne peut pas resoudre

BiRefNet fait de la **detection d'objet saillant**. Il repond a « qu'est-ce qui est au premier
plan ? ». Il n'a aucune notion d'identite d'objet.

Or sur les photos reelles, la piece est **suspendue a une pince de levage par une elingue**, posee
au contact d'un etabli (`piece_test/capture.json`). La pince est au premier plan, elle touche la
piece, elle est nette et contrastee. BiRefNet la garde — et il a raison au regard de sa tache.

SAM fait de la **segmentation d'instance promptable** : « quel objet, sachant cet indice ? ». Un
clic sur la piece suffit a exclure la pince, et un clic droit sur la pince l'exclut explicitement.
C'est structurellement une autre question, pas un meilleur detourage.

## Place dans la chaine

```text
photo  ->  BiRefNet  ->  point d'amorce  ->  SAM 3  ->  PNG RGBA  ->  Pixal3D
              (facultatif, generateur de prompt)          ^
                                                          |
                                   ou simplement le clic de l'utilisateur
```

**BiRefNet n'est pas une premiere passe de nettoyage.** Son masque ne sort jamais du systeme : il
sert uniquement a choisir ou cliquer quand personne ne clique. Le masque de SAM fait foi. Sans
cette regle on empile deux masques sans savoir lequel gagne.

## Pourquoi le PNG RGBA suffit

`preprocess_image` (`vendor/pixal3d/pixal3d/pipelines/pixal3d_image_to_3d.py`) teste l'alpha :

```python
has_alpha = False
if input.mode == 'RGBA':
    alpha = np.array(input)[:, :, 3]
    if not np.all(alpha == 255):
        has_alpha = True
...
if has_alpha:
    output = input          # rembg n'est jamais appele
```

Fournir un RGBA dont l'alpha n'est **pas uniformement 255** court-circuite donc proprement le
detourage amont. Aucune modification de `vendor/` (contrainte n°6).

Deux consequences inscrites dans le code :

- `compose_rgba` **refuse** un masque plein. Un alpha tout a 255 serait ignore et Pixal3D
  relancerait son propre detourage, ramenant la pince. L'erreur tombe sur CPU, pas sur GPU facture.
- `save_rgba` **refuse** une extension autre que `.png`. Un JPEG n'a pas de canal alpha.

## Le piege du prompt automatique

Le centroide du masque BiRefNet **echoue sur cette piece**. C'est une roue avec alesage central :
le centroide tombe dans le trou, donc dans le fond, et SAM segmente l'etabli.

`core/segmentation.py` utilise donc le **maximum de la transformee de distance** a l'interieur du
masque — le centre du plus grand cercle inscrit. Sur un disque il coincide avec le centroide ; sur
un anneau il se place sur le rayon median. C'est aussi le point ou un humain cliquerait.

Le test `test_the_centroid_of_an_annulus_falls_in_the_hole_which_is_why_it_cannot_be_the_prompt`
verrouille le constat, et `largest_connected_component` ne bouche jamais les trous : remplir
l'alesage remettrait le point d'amorce au milieu du fond.

## Modele et licence

| | |
|---|---|
| Checkpoint | `facebook/sam3`, **gated** (`license: other`) |
| Classe pour le clic | `Sam3TrackerModel` / `Sam3TrackerProcessor` |
| Parametres | 0,9 Md, ~3,4 Go de poids |
| VRAM mesuree en doc amont | < 4 Go en inference |
| GPU local | RTX 4070 Laptop, 8188 Mio, sm_89 — suffisant |

**Le clic n'est pas `Sam3Model`.** La nouveaute de SAM 3 est la segmentation par *concept* (texte
ou exemplar) et `Sam3Model` n'expose que `text` et `input_boxes` — aucun `input_points`. Un clic
utilisateur releve de la segmentation visuelle promptable, portee par `Sam3TrackerModel`, decrit
en amont comme « SAM 2 avec la meme API » et de meilleurs poids. Meme checkpoint, autre tete.

La licence gated est un revirement assume par rapport a ADR-0010 : voir **ADR-0014**.

## Installation

```powershell
uv sync --extra gui --extra segmentation
```

L'extra `segmentation` n'est **pas** dans le groupe `dev` : il pese ~3 Go et le gate CPU doit
rester installable en CI gratuite. `backends/sam3.py` importe `torch` et `transformers`
paresseusement, dans `load()`.

Le jeton Hugging Face est lu, dans l'ordre : `HUGGINGFACE_TOKEN`, `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, puis le fichier `huggingface.env` (jeton nu sur une ligne, ou
`NOM=valeur`). Ce fichier est couvert par `.gitignore` (`*.env`).

**Aucun message d'erreur ne contient jamais le jeton** — un 401 du Hub renvoie l'en-tete
`Authorization` en clair, et il finirait dans `runs/<id>/logs.jsonl`. Le test
`test_no_error_message_ever_contains_the_token` injecte exactement cette fuite pour prouver que la
redaction fonctionne.

## Chargement paresseux

Ouvrir l'onglet « Découpe (SAM 3) » **ne construit rien**. Le runner est bati au premier clic,
via `runner_factory`. `test_opening_the_window_does_not_touch_the_model` verifie que
`has_engine` reste faux apres l'ouverture de la fenetre.

## Ce que F14 n'apporte pas

- **Elle n'avance pas le gate F13.** Le benchmark synthetique rend sur fond noir avec masques
  exacts : la segmentation n'y joue aucun role. F14 sert le chemin photos reelles, et
  plausiblement le produit final, ou l'utilisateur tapera sur la piece a l'ecran.
- **Le backend BiRefNet n'existe pas encore** dans `backends/`. Le patch vit toujours dans
  `/workspace/run_mit.py` sur le volume RunPod (ADR-0010). `prompt_automatically` accepte donc
  n'importe quel masque grossier ; brancher BiRefNet reste a faire.
- **Aucune verification sur GPU reel.** Toute la verification F14 substitue le modele. La
  qualite des masques de SAM sur pieces metalliques mates au contact d'outillage metallique
  n'est pas mesuree, et c'est precisement le cas difficile.

## Verification F14

```powershell
uv run pytest -q tests/unit/test_segmentation.py tests/unit/test_sam3_backend.py `
  tests/integration/test_gui_segmentation.py
```
