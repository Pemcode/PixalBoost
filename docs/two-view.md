# Deux vues sans calibration

> Contrat F15. A lire avant de toucher a `core/pose_search.py`,
> `trials/two_view.py` ou `gui/two_view_view.py`.

## Le probleme

Pixal3D est **pixel-aligne** et `camera_params` se reduit a
`{camera_angle_x, distance, mesh_scale}` : **aucune extrinseque**. Deux photos donnent donc deux
objets dans deux orientations differentes, chacun vu depuis la meme camera canonique. Rien dans
la sortie ne dit comment ils se rapportent l'un a l'autre.

La reponse habituelle est de calibrer : une mire ArUco dans la scene, ou un plateau indexe. Les
deux imposent de refaire la prise de vue.

## La reponse : l'objet est sa propre mire

On cherche la rotation de l'objet dont la **silhouette rendue** maximise l'IoU avec le **masque
de la seconde photo**. Le rasteriseur est celui de F03 (`core/render.py`) : CPU pur, pas de GPU,
pas de modele, pas de dependance nouvelle.

```
reconstruction A  --rendu--> silhouette(R)  ~=  masque SAM de la photo B
                              ^
                        on cherche R
```

Trois proprietes le rendent viable la ou l'appariement de descripteurs echouerait.

**Les silhouettes survivent au metal.** Une piece moulee mate n'offre presque aucun detail
photometrique repetable, et ses reflets se deplacent avec la camera. Son contour, non.

**Ca marche quand la piece tourne, pas la camera.** VGGT ou MASt3R deduisent la pose du mouvement
de la scene : si l'on tourne la piece a la main devant un atelier immobile, ils voient une camera
fixe et ne trouvent rien. Le rendu-comparaison s'aligne sur **l'objet** ; ce qui bouge autour lui
est indifferent. En atelier on tourne la piece.

**ADR-0007 s'inverse.** Cet ADR refuse une pose ambigue parce qu'une pose fausse etale une
fusion sans rien signaler. Ici la pose ne sert qu'a *rendre*, et une rotation autour d'un axe de
symetrie reel laisse le rendu inchange : l'ambiguite est **reportee et inoffensive**. Le test
`test_an_axisymmetric_part_is_reported_ambiguous_yet_still_renders_correctly` l'exige.

## Le recadrage, sans lequel tout est faux

`preprocess_image` recadre **chaque vue sur sa propre bbox alpha**, en carre, x1.1. Comparer un
masque brut a un rendu canonique, c'est comparer deux cadrages differents : les silhouettes se
recouvrent quand meme beaucoup, la recherche converge, et la reponse est confiante et fausse.

`crop_to_canonical_framing` applique la meme regle **aux deux cotes**. La recherche porte alors
sur la rotation seule : l'echelle et le centrage sont absorbes au lieu d'etre ajustes.

Une deviation assumee : l'etendue est mesuree en *nombre de pixels* (`max - min + 1`) et non en
difference d'indices comme en amont. Cela rend le cadrage exactement invariant d'echelle, ce que
la version en indices n'est pas, au prix d'un pixel sur une photo de 4000.

## Ce que le GLB contient

**Les deux reconstructions dans un repere commun. Rien de fusionne.** Le nombre de faces est
exactement additif et un test l'exige. La contrainte n°11 reste entiere jusqu'au gate F13.

L'artefact repond a une seule question, et elle se lit a l'oeil : **les deux moities se
superposent-elles ?**

## Les deux nombres, et pourquoi il en faut deux

| Metrique | Ce qu'elle dit |
|---|---|
| `pose_iou` | le rendu tourne colle au masque de la photo B |
| `agreement_iou` | une fois alignees, les deux reconstructions se recouvrent |

**`pose_iou` seul ne prouve rien.** Mesure : un masque en barre score **0,94** contre un
L-bracket — qui vu de chant *est* une barre. La silhouette ne distingue pas un vrai accord d'une
vue degeneree qui coincide. Le test
`test_a_bar_shaped_mask_does_NOT_collapse_the_score_and_that_is_a_real_limit` epingle la limite
pour que personne ne la redecouvre sur de vraies photos.

En revanche un masque **topologiquement impossible** — un anneau contre un solide sans trou —
fait bien s'effondrer le score.

## Utilisation

Onglet **Reconstruction 2 vues** de `uv run pixaboost-gui` : deux champs, deux `Parcourir…`, un
bouton. Le bouton reste inactif tant que les deux photos ne sont pas choisies, refuse deux fois
la meme image, et refuse un fichier absent — le tout avant de depenser quoi que ce soit.

Prends **face avant et face arriere**, pas deux vues voisines : la pose se deduit d'autant mieux
que les deux silhouettes different.

## Ce que F15 ne fait pas

- **Aucune fusion, aucune sculpture, aucun score de confiance par voxel.** Contrainte n°11.
- **Aucune echelle absolue.** Une cote au pied a coulisse la fixe ; rien de monoculaire ne le fait.
- **Aucun bout en bout verifie.** Reconstruire chaque photo exige un Pod actif, donc F07. Aucun
  GLB n'a encore ete produit depuis de vraies photographies.

## Verification F15

```powershell
uv run pytest -q tests/unit/test_pose_search.py tests/unit/test_two_view_trial.py `
  tests/integration/test_gui_two_view.py
```
