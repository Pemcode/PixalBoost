# Internes de Pixal3D

> Livrable du spike F01. A lire avant de toucher a `backends/pixal3d.py`.
> Submodule pinne sur `cdbb2bb` (2026-06-23). Les assertions de ce document sont
> executables : `uv run pytest tests/unit/test_pixal3d_contract.py`.

## Verdict

**Non. Le chemin multi-vues n'est pas expose — et il est activement bloque.**

Le parametre de camera est cable de bout en bout dans la chaine de conditionnement, mais un
`assert` en fin de course impose qu'il soit `None` :

```python
# vendor/pixal3d/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:211
assert transform_matrix is None, "transform_matrix is not None"
```

Toute back-projection est donc forcee sur une **camera front-view canonique** dont seule la
distance varie (`transform_matrix[:, 1, 3] = -distance`, ligne 215).

### Ce que « 2 views by default » signifie reellement

Le README amont pouvait laisser croire a un support multi-vues. Ce n'est pas le cas : c'est de
l'**echantillonnage a l'entrainement**. Les datasets disposent de 2 rendus par objet et en tirent
**un seul** au hasard a chaque pas :

```python
# pixal3d/datasets/structured_latent_shape.py:245  -> num_views: int = 2
# pixal3d/datasets/structured_latent_shape.py:330  -> view_idx = np.random.randint(0, self.num_views)
```

C'est ce qui rend le modele *pixel-aligne* — il apprend a produire dans le repere de la vue qu'on
lui donne, quelle qu'elle soit. Ce n'est pas un conditionnement sur plusieurs vues simultanees.

## Preuves

| Fait | Emplacement |
|---|---|
| `assert transform_matrix is None` bloque toute camera arbitraire | `image_conditioned_proj.py:211` |
| Camera front-view fixe, seule la distance varie | `image_conditioned_proj.py:172-178`, `:215` |
| `transform_matrix` est pourtant cable jusqu'en haut | `:1282` (`encode_image_proj`), `:470` (`DinoV3ProjFeatureExtractor.forward`) |
| La projection accepte deja une matrice `[B,4,4]` arbitraire | `image_conditioned_proj.py:27-108` |
| `valid_mask` est calcule puis **jete** | calcule `:100-104`, lie `:218`, jamais relu |
| Hors-champ = features de bord, pas exclusion | `sample_features`, `padding_mode='border'` `:132` |
| `B` est la dimension de **batch d'objets**, pas de vues | `:500`, `:520` |
| Le pipeline fige `B = 1` | `pixal3d_image_to_3d.py:267`, reshape `:275` |
| La CLI ne prend qu'une image | `inference.py:288` |

## Modele de camera (a respecter dans `core/geometry.py`)

Convention **Blender**, pas OpenCV. A convertir a la frontiere, dans `backends/`.

- La camera regarde selon **-Z** : `depth = -z_cam` (`image_conditioned_proj.py:81`).
- Capteur 32 mm, focale `16.0 / tan(camera_angle_x / 2)`, puis `* resolution / 32`
  (`:84-86`, repris dans `inference.py:118-121`).
- La grille 3D est `grid_resolution³` points dans `[-1, 1]`, tournee par
  `[[1,0,0],[0,0,-1],[0,1,0]]` (`:162-166`, meme matrice dans `inference.py:125`).
- Les points sont divises par `mesh_scale` puis par 2 (`:210`). L'objet vit dans une AABB
  `[-0.5, 0.5]³` a l'export (`inference.py:266`).
- `camera_params` se reduit a `{camera_angle_x, distance, mesh_scale}` (`inference.py:137-155`) :
  **il n'y a pas d'extrinseque**. L'orientation de l'objet est portee par la generation elle-meme.

Consequence directe pour la fusion : deux vues produisent deux objets dans deux orientations
differentes, chacun vu depuis la meme camera canonique. **Le recalage inter-vues est donc
obligatoire, il ne peut pas etre lu depuis la sortie.**

## Cascade et cout

Trois etages, tous conditionnes par back-projection DINOv3
(`camenduru/dinov3-vitl16-pretrain-lvd1689m`, `inference.py:26-53`) :

| Etage | Grille | Resolution image | Branche NAF |
|---|---|---|---|
| Sparse Structure (`ss`) | 16 | 512 | non |
| Shape 512 / 1024 | 32 / 64 | 512 / 1024 | oui |
| Texture 1024 | 64 | 1024 | oui (cible 1024) |

Depth camera par **MoGe-2** (`Ruicheng/moge-2-vitl`), uniquement pour estimer le FOV puis
dechargee (`inference.py:212-224`). Detourage par BiRefNet via `pipeline.preprocess_image`.
Export GLB par `o_voxel.postprocess.to_glb`, texture 4096, decimation 1 M
(`inference.py:263-269`).

**VRAM** : ~18 Go en mode standard, ~10-12 Go en `--low_vram` (`inference.py:296-298`).
Resolution 1536 par defaut, 1024 en low-VRAM.

## Ce que F10 doit faire

La bonne nouvelle : la plomberie mathematique existe deja. `project_points_to_image_batch`
accepte une `transform_matrix [B,4,4]` quelconque. L'intervention est **chirurgicale**, pas une
reecriture.

1. Sous-classer `ProjGrid` dans `backends/pixal3d.py` pour lever l'`assert` et **retourner
   `valid_mask`** au lieu de le jeter.
2. Ecrire l'agregateur multi-vues : une passe par vue avec sa propre `transform_matrix`, puis
   **moyenne masquee** par voxel. Sans le masque, les voxels hors-champ polluent la moyenne avec
   des features de bord.
3. Dupliquer `get_proj_cond_ss` / `get_proj_cond_shape` en variantes prenant N images et N
   transformations, et lever le `B = 1`.
4. Fournir les poses relatives exprimees dans le repere canonique Blender ci-dessus.

Rappel de la contrainte n°6 de `CLAUDE.md` : rien de tout cela ne s'edite dans `vendor/`.

## Risque n°1 — le checkpoint publie est entraine en mono-vue

Les poids ont ete entraines avec un conditionnement **mono-vue a camera fixe** : un seul
`view_idx` tire par pas, `transform_matrix` toujours `None`. Un volume de features **moyenne sur
plusieurs vues** est donc une entree **hors distribution** pour ce checkpoint.

Le papier annonce que la methode s'etend naturellement au multi-vues, mais rien dans le code
publie ne prouve que *ces poids-la* ont vu ce cas. Il est possible que les resultats multi-vues
du papier viennent d'une variante non publiee.

**Consequence sur le plan** : l'hypothese H1 (« B1 bat B0 ») est nettement plus fragile qu'anticipe.
F12 doit donc mesurer B1 **et** un garde-fou : si la moyenne multi-vues degrade la sortie, tester
la variante « une seule vue conditionne, les autres ne servent qu'au recalage et a la texture »
avant de conclure. A trancher au gate F13, avec les chiffres.
