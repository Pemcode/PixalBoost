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

## Le prior « faces opposees » — le bit que la silhouette ne porte pas

**Mesure sur `piece_test/front_back`** : les deux vues de la roue ont la **meme silhouette**, un
cercle. La geometrie interne differe nettement — moyeu saillant d'un cote, cuvette de l'autre —
mais le *contour* est identique. Aucune recherche par silhouette ne peut donc distinguer l'avant
de l'arriere ; elle renvoie l'identite, et les deux moities se superposent au lieu de se completer.

Or **le photographe sait** que la seconde vue est le dos. C'est un bit d'information qu'il possede
toujours gratuitement et que l'image ne contient pas. `OPPOSITE_FACES` le fait entrer comme
contrainte : la recherche est confinee a +/- 70 deg d'un demi-tour, ce qui **affirme** le
retournement au lieu de le deviner, et laisse la silhouette determiner l'inclinaison — la seule
chose qu'elle sache vraiment determiner.

La case est **cochee par defaut** dans la GUI, parce que c'est le cas etiquete. Decoche-la si la
seconde vue est un autre angle plutot qu'un retournement.

**Ce que le prior ne repare pas.** Il *reduit* la degenerescence sans la supprimer : sur le cas
du L-bracket vu de chant, le score tombe de 0,94 a 0,83 — et `is_trustworthy` reste vrai sur un
masque qui n'a rien a voir avec la piece. Un test l'exige explicitement, pour que ce ne soit pas
oublie.

## Utilisation

Onglet **Reconstruction 2 vues** de `uv run pixaboost-gui`.

**L'entree est une decoupe RGBA, pas une photo.** Passe d'abord par l'onglet
« Decoupe (SAM 3) » et enregistre `<nom>_cutout.png` pour chacune des deux vues. Le canal alpha
de ces fichiers est **le masque qui determine la pose** ; un JPEG est refuse en nommant le
fichier fautif. La raison est en ADR-0016 : le masque juge doit etre le masque utilise.

Prends **face avant et face arriere**, pas deux vues voisines : la pose se deduit d'autant mieux
que les deux silhouettes different.

### Ce qui se passe quand tu cliques

1. **Preflight local, gratuit.** Les deux alphas sont lus (un mauvais fichier coute zero) et le
   cache d'artefacts est interroge pour chaque vue.
2. **Confirmation, seulement s'il manque quelque chose.** La boite nomme les vues absentes du
   cache et le nombre de reconstructions que cela represente. Un cache complet ne demande rien.
3. **Reconstruction, une approbation fraiche par photo.** `ExistingPodUseApproval` est a usage
   unique et expire en 120 s alors qu'une reconstruction dure des dizaines de minutes ; une seule
   approbation serait refusee sur la seconde vue, apres avoir paye la premiere.
4. **Recherche de pose, puis ecriture** de `runs/<id>/aligned.glb` et de son manifeste.

Le bouton **Annuler** est actif pendant tout l'essai : une reconstruction pas encore commencee
n'est jamais achetee. Si l'etat du Pod ne peut pas etre confirme, le panneau le dit au lieu de se
taire. Fermer la fenetre pendant un essai est refuse — les threads sont enfants du panneau, et
une reconstruction en cours est du temps GPU facture.

**Un seul essai a la fois.** L'essai local, l'essai mono-vue et l'essai 2 vues s'excluent
mutuellement : deux sessions SSH sur le meme GPU entrelaceraient deux reconstructions et
factureraient les deux.

## Ou vit quoi

| Module | Role |
|---|---|
| `core/pose_search.py` | la recherche elle-meme, CPU pur, deterministe |
| `core/segmentation.py` | `mask_from_rgba` — l'alpha relu au seuil de Pixal3D |
| `backends/images.py` | lecture du fichier, EXIF applique ; aucune logique |
| `backends/glb.py` | chargement et ecriture des maillages places |
| `trials/two_view.py` | orchestration + manifeste ; `reconstruct` et `mask_of` sont **injectes** |
| `gui/two_view_adapter.py` | le cablage reel : cache-first, approbation, annulation |
| `gui/two_view_view.py` | le panneau ; ne decide jamais de depenser |

## Ce que F15 ne fait pas

- **Aucune fusion, aucune sculpture, aucun score de confiance par voxel.** Contrainte n°11.
- **Aucune echelle absolue.** Une cote au pied a coulisse la fixe ; rien de monoculaire ne le fait.
- **Aucun bout en bout verifie.** Le moteur est cable et teste transport substitue, mais
  reconstruire chaque photo exige un Pod actif, donc F07. **Aucun GLB n'a encore ete produit
  depuis de vraies photographies.**

## Verification F15

```powershell
uv run pytest -q tests/unit/test_pose_search.py tests/unit/test_two_view_trial.py `
  tests/unit/test_two_view_adapter.py tests/unit/test_images_backend.py `
  tests/unit/test_segmentation.py tests/integration/test_gui_two_view.py
```
