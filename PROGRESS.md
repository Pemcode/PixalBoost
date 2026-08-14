# PROGRESS

> Journal d'etat entre sessions. La session qui arrive lit ce fichier en premier ;
> la session qui part le met a jour avant de commiter. Voir la routine dans `CLAUDE.md`.

## Etat verifie

- **Sprint 0 : transport Pod et GUI d'essais livres.** F00 a F06, F08 et F09 sont `passing` ;
  F07 reste `blocked`.
- **Sprint 1 : entame.** F11, F14 et F15 sont `passing`.
- **Dernier gate vert** : `uv run poe check` — ruff + mypy strict + **408 tests**,
  **20,63 s** le 2026-08-13. Aucun GPU ni reseau.

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
| F08 GUI d'essais | `passing` | 73 tests cibles + vrai benchmark CPU observe |
| F09 transport ssh-pod | `passing` | 74 tests, ADR-0012 et ADR-0013 |
| F11 recalage | `passing` | 25 tests, ADR-0006 et ADR-0007 |
| F14 segmentation par clic | `passing` | 86 tests, ADR-0014, **modeles substitues partout** |
| F15 pose 2 vues sans calibration | `passing` | 103 tests, moteur cable ; **jamais teste sur vraies photos** |

## Le jalon de la session du 2026-08-13

**Une GUI PyQt6 d'essais CPU observables est livree.** Lancement :
`uv run pixaboost-gui`.

- trois commandes gratuites et locales : gate, tests et benchmark synthetique ;
- etat, phase, progression disponible, commande sanitisee, stdout/stderr, duree, code retour et
  inventaire des artefacts visibles en direct ;
- processus execute hors du fil graphique, single-flight et arret de l'arbre local sous Windows ;
- protocole de telemetrie JSONL type et teste, sans coupler `core/` a Qt ;
- succes refuse si un manifeste requis manque, est perime ou n'est pas un objet JSON valide ;
- reconstruction GPU explicitement desactivee tant que F07 ne fournit pas de transport Pod.

La verification F08 compte 73 tests verts. Le controleur a aussi lance la vraie commande de
benchmark par defaut : 58 evenements, succes en 3,97 s et manifeste produit. Aucun appel GPU,
aucun reseau payant et aucune operation sur des credits n'ont ete effectues.

## F14 — segmentation par clic (SAM 3)

**Le probleme reel** : sur les photos, la piece est suspendue a une **pince de levage** par une
elingue, posee au contact d'un etabli. BiRefNet garde la pince, et il a raison : il fait de la
detection d'objet saillant, il repond a « qu'est-ce qui est au premier plan ? », sans notion
d'identite d'objet. SAM fait de la segmentation d'instance promptable — « quel objet, sachant cet
indice ? ». C'est structurellement une autre question, pas un meilleur detourage.

Chaine : **BiRefNet -> point d'amorce -> SAM 3 -> PNG RGBA -> Pixal3D**. BiRefNet est demote en
*generateur de prompt* ; son masque ne sort jamais du systeme. Le masque de SAM fait foi.

- Le PNG RGBA court-circuite proprement le detourage amont : `preprocess_image` n'appelle pas
  rembg si l'alpha n'est pas uniformement 255. `compose_rgba` **refuse** un masque plein pour
  cette raison — l'erreur tombe sur CPU, pas sur GPU facture. Aucune modification de `vendor/`.
- Le prompt automatique est le **maximum de la transformee de distance**, pas le centroide : la
  piece est une roue a alesage central et son centroide tombe dans le trou, donc dans le fond.
- Le clic passe par `Sam3TrackerModel`, pas `Sam3Model` : la nouveaute de SAM 3 est le prompt
  *textuel*, et `Sam3Model` n'expose aucun `input_points`.
- Ouvrir l'onglet ne telecharge rien ; le modele est construit au premier clic.

### Ce que la revue de GUI a trouve

Les tests etaient verts et la feature etait **inutilisable**. Trois defauts, tous invisibles aux
tests parce qu'ils appelaient les methodes directement au lieu de passer par l'interface :

1. **Le bouton d'enregistrement n'etait connecte a rien.** `undo` et `reset` l'etaient, pas lui.
2. **Aucun moyen de charger une photo.** L'onglet s'ouvrait sur « aucune image » sans issue.
3. **BiRefNet n'etait pas cable du tout.** `prompt_automatically` existait, teste, et personne
   ne l'appelait ; aucun backend de saillance n'existait.

Corriges, plus `backends/birefnet.py` (MIT, non gated, ADR-0010), l'application de l'EXIF au
chargement, des marqueurs de points, et un apercu qui **assombrit le fond** au lieu de teinter la
selection — la teinte etait invisible, l'accent du theme etant bleu comme la piece.

Lecon a retenir : *un test qui appelle `panel.save_rgba(path)` ne prouve rien sur le bouton.*

### Premiere mesure sur poids reels — 2026-08-13

`facebook/sam3` charge et execute en local sur `piece_test/upright/view01.jpg` (2048 x 1153),
CPU, quatre points d'amorce. **La these de F14 est verifiee : SAM isole la roue et laisse la
sangle de levage dehors.**

| Clic | Couverture | Score SAM | Masque obtenu |
|---|---:|---:|---|
| jante droite | **19,4 %** | 0,976 | **roue entiere, sangle exclue** |
| jante gauche | **19,3 %** | 0,951 | **roue entiere, sangle exclue** |
| haut de roue (sur la sangle) | 1,8 % | 0,967 | la sangle seule |
| centre-bas (dans l'alesage) | 1,4 % | 0,953 | quasi rien |

Trois enseignements, tous mesures :

1. **Le piege du centroide est reel.** Le clic tombe dans l'alesage rend 1,4 %. C'est exactement
   ce que `deepest_interior_point` evite, et la raison d'etre de `core/segmentation.py`.
2. **Le score de SAM ne mesure pas la justesse.** Les quatre resultats sont entre 0,95 et 0,98,
   y compris les deux mauvais. Il dit « ce masque est net », pas « c'est la bonne piece ». La GUI
   affiche donc desormais la **couverture** en premier et alerte sous 4 %.
3. **Cout CPU** : ~19 s par clic sur 2048 px, chargement ~40 s depuis le cache. Utilisable, mais
   la version CUDA s'impose pour un usage interactif.

**Ce que F14 ne prouve toujours pas.** BiRefNet n'a jamais tourne pour de vrai (seul SAM 3 l'a
fait). Une seule photo, une seule piece, aucune metrique contre une verite terrain — l'evaluation
est visuelle. Et F14 **n'avance pas le gate F13**.

Licence gated assumee en **ADR-0014**, explicitement contre le precedent d'ADR-0010. La condition
de reexamen y est ecrite pour etre falsifiable.

## F15 — deux vues sans calibration

**L'objet est sa propre mire.** Pixal3D est pixel-aligne et ne rend aucune extrinseque : deux
photos donnent deux objets dans deux reperes sans relation. Plutot que de poser une mire ou un
plateau indexe, on **derive** la pose : on cherche la rotation dont la silhouette rendue de la
reconstruction A colle au masque de la photo B.

- Le rasteriseur de F03 fait tout le travail. Aucun GPU, aucun modele, aucune dependance nouvelle.
- **Les silhouettes survivent au metal** : un appariement photometrique echouerait sur une piece
  mate a reflets mobiles ; un contour non.
- **ADR-0007 s'inverse.** Il refuse une pose ambigue parce qu'une pose fausse etale une fusion.
  Ici la pose ne sert qu'a *rendre*, et une rotation autour d'un axe de symetrie reel ne change
  pas le rendu : l'ambiguite est signalee et ne coute rien. Un test l'exige explicitement.
- Le masque de la photo passe par **le meme recadrage que `preprocess_image`** (carre autour de
  la bbox, x1.1). Sans cette etape on compare deux cadrages differents et la recherche converge
  sur une reponse confiante et fausse — c'etait le « tueur silencieux » identifie dans le vendor.

**Le GLB ne fusionne rien.** Il juxtapose les deux reconstructions dans un repere commun ;
le nombre de faces est exactement additif, et un test l'exige. La contrainte n°11 reste entiere.
L'artefact sert a repondre a une seule question : les deux moities se superposent-elles ?

**Limite mesuree et epinglee** : un masque en barre score 0,94 contre un L-bracket — qui vu de
chant *est* une barre. La silhouette seule ne distingue pas un vrai accord d'une vue degeneree.
`pose_iou` ne prouve donc rien seul ; `agreement_iou` et l'inspection du GLB sont les garde-fous.
Un masque topologiquement impossible (un anneau) fait bien s'effondrer le score.

### Le moteur est cable (2026-08-14)

Le bouton n'est plus un ecriteau. `gui/two_view_adapter.py` enchaine deux
`run_single_view_trial` cache-first sur un Pod existant, et **le masque vient du canal alpha des
decoupes RGBA** produites par l'onglet SAM 3 (ADR-0016) : un seul fichier porte l'image que
Pixal3D reconstruit *et* la silhouette contre laquelle la pose est cherchee, donc ils ne peuvent
pas diverger.

Les assertions qui comptent sont negatives — chacune est un enonce sur l'argent :

| Ce qui est verifie | Ce que ca coute quand ca se produit |
|---|---|
| Un JPEG est refuse **en le nommant**, dans le preflight local | zero |
| Un cache miss sans approbation ne construit **aucun** client | zero |
| Un second essai identique n'achete rien (contrainte n°9) | zero |
| Une **approbation fraiche par photo** — l'octroi est a usage unique et expire en 120 s, une reconstruction dure des dizaines de minutes | sinon : la seconde vue est refusee **apres** avoir paye la premiere |
| Une annulation empeche la seconde reconstruction | ~25 min de 4090 |
| Un seul essai a la fois : local, mono-vue et 2 vues s'excluent | sinon : deux sessions SSH sur le meme GPU, facturees toutes les deux |

**Deux defauts trouves pendant le cablage, tous deux invisibles en une seule execution du gate.**
Le premier : un `QThread` d'annulation survivait a son test, et Qt abandonnait le processus —
le gate echouait environ **une fois sur six**, sans aucun rapport d'echec, dans un test different
a chaque fois. Le panneau expose maintenant `is_busy`, les tests l'attendent, et
`MainWindow.closeEvent` refuse de fermer par-dessus un thread vivant. Le second : le message
« etat du Pod inconnu » etait ecrase par le resultat du run selon l'ordre d'arrivee des deux
signaux ; il est desormais reapplique quel que soit l'ordre, et volontairement **pas** ajoute a
un run reussi — les deux reconstructions ont fini, le Pod ne travaille plus.

**Toujours pas de bout en bout reel** : le transport est substitue dans tous les tests. Ce qui
manque est un Pod actif, donc F07.

## Ce que la revue de cloture du 2026-08-13 a corrige

Le travail livre etait fonctionnellement bon — 315 tests verts — mais **trois affirmations du
depot etaient fausses**, et c'est le genre de defaut que le harness existe pour attraper.

1. **~2 300 lignes livrees sans commande de verification.** `backends/ssh_pod.py`,
   `backends/ssh_worker_source.py`, `trials/`, `gui/remote_trial.py` et
   `gui/single_view_adapter.py` n'avaient **aucune entree** dans `feature_list.json`. Elles
   etaient couvertes par le gate global, mais aucune commande ne disait ce qu'elles devaient
   prouver. -> **F09** ouverte retroactivement, verification executee : 74 tests verts en 2,93 s.
2. **Deux commandes de verification F08 en circulation.** `docs/gui.md` en declarait une
   (5 fichiers, 73 tests), `feature_list.json` une autre (2 fichiers). Celle de `docs/gui.md`
   n'avait jamais tourne. -> executee, verte, et adoptee comme reference.
3. **`DECISIONS.md`, `README.md` et `feature_list.json` se contredisaient sur F07.** ADR-0011
   annoncait le transport comme livre par F07 ; la note de F08 le declarait hors scope ; F07
   restait `blocked` sur « il manque un transport SSH » alors que le fichier existait. -> les
   trois recalees sur F09, et le vrai blocage de F07 enonce : le test e2e vise encore le client
   serverless, et aucune execution GPU n'a eu lieu.

Garde-fous reverifies par **injection d'une violation reelle**, pas par lecture :
`test_architecture.py` se declenche bien quand on ecrit `import torch` et
`from pixaboost.gui...` dans `core/geometry.py` ; `test_harness.py` se declenche sur les quatre
violations possibles de `feature_list.json` (JSON casse, deux features `active`, `passing` sans
preuve, `blocked` sans cause). Contraintes n°4 et n°11 intactes : `core/` reste sans torch,
sans reseau et sans code de fusion.

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

**F07** — le transport SSH existe (F09). Ce qui manque est l'**execution reelle** :
`tests/e2e/test_smoke_single_view.py` vise encore le client serverless (`RUNPOD_ENDPOINT_ID`) et
non le `SshPodClient` effectivement construit. Il faut reecrire le test sur le backend `ssh-pod`,
puis le lancer sur un Pod actif. C'est le seul travail restant sur F07.

## Correction : le push n'est PAS bloque

Cette section affirmait `! [remote rejected] main -> main (pre-receive hook declined)` et
« 1 commit en attente ». **Verifie le 2026-08-13 : c'est faux.**

```
git ls-remote origin refs/heads/main -> b9655e4cbb5cec0e9388f0fd1b56d82186ba4646
git rev-parse main                   -> b9655e4cbb5cec0e9388f0fd1b56d82186ba4646
```

Le correctif natten (`ee4a058`) est un ancetre de `b9655e4` : **il est sur GitHub**. Le blocage
decrit ici a ete resolu sans que la note soit mise a jour, et deux sessions ont pu partir d'un
diagnostic perime.

**Confirme le 2026-08-13** : l'image publiee **porte le correctif**. Les tags GHCR incluent
`gpu-b9655e4cbb5cec0e9388f0fd1b56d82186ba4646`, et `gpu-latest` a **le meme digest**
(`sha256:bc094b3f4cf4…`). Or `ee4a058` est un ancetre de `b9655e4`. L'ancienne image cassee
etait `gpu-5ff20035…` (digest `sha256:6bbf9f0ad1c5…`), differente.

Consequence : **`bash /workspace/setup_pod.sh` ne devrait plus etre necessaire** sur un pod
demarre depuis `gpu-latest`. A confirmer une fois sur le pod, mais l'indice est fort.

## Correctifs encore manuels dans le pod

Ils vivent sur le volume reseau et survivent aux redemarrages :

- `/workspace/setup_pod.sh` — reinstalle le natten officiel (ADR-0009). **Devenu inutile** :
  `gpu-latest` est construit sur `b9655e4`, qui contient le correctif. Verifie par digest GHCR.
- `/workspace/run_mit.py` — force BiRefNet MIT au lieu de RMBG-2.0 gated (ADR-0010). Doit etre
  porte proprement dans `backends/`.

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
1. ~~Confirmer que l'image publiee porte le correctif natten~~ — **fait**, digest GHCR verifie.
2. ~~Cabler le moteur 2 vues~~ — **fait** le 2026-08-14, transport substitue.
3. **Le premier essai reel 2 vues.** Il faut, dans cet ordre : decouper les deux photos de
   `piece_test/front_back/` dans l'onglet SAM 3 (elles sont en JPEG, donc refusees telles
   quelles), un Pod actif avec hote/port/cle, puis un clic. Compter ~50 min de 4090 pour les
   deux vues. Le prior « faces opposees » est indispensable ici : les deux silhouettes de cette
   roue sont le meme cercle.
4. Reecrire `tests/e2e/test_smoke_single_view.py` sur `SshPodClient` et l'executer : c'est la
   seule chose qui debloque F07 formellement.
5. Porter le patch BiRefNet dans `backends/` plutot que dans un script sur le volume.
6. Batcher l'inference — prerequis economique de F12.
7. F10, le chemin multi-vues.

## Jeu de photos reelles disponible

18 photos d'une **piece metallique mate** sont versionnees sous `piece_test/view01..18.jpg`,
avec `piece_test/capture.json` qui documente honnetement ce qu'on sait et ce qu'on ignore.

- **Aucune pose camera mesuree.** La piece etait suspendue a une pince de levage : son orientation
  variait entre les prises. Le protocole « 6 azimuts x 3 elevations » decrit l'intention, pas la
  donnee. Ne jamais les reprendre comme initialisations nominales pour F10.
- **La piece est axisymetrique** (roue/moyeu, repere 249, alesage central) — exactement le cas que
  le score d'ambiguite de F11 refuse. ADR-0007.
- **Defauts connus** : EXIF `Orientation=6` sur les 18 (PIL ne l'applique pas) ; pince et elingue
  **en contact** avec la piece ; fonds d'atelier charges.
- Les derives (`upright/` EXIF applique + 2048 px, `preview/` PNG) sont ignores par git : contenu
  recalculable, ~18 Mo d'incompressible. Un script de regeneration reste a ecrire.

## Notes

- `docker`, `make` et `gh` ne sont pas installes en local (Windows). Assume.
- **GPU local disponible : RTX 4070 Laptop 8 Go (sm_89).** Compatible avec la stack torch 2.6/cu124
  (ADR-0008), mais 8 Go ne suffisent pas a la cascade Pixal3D complete. Utilisable pour les modeles
  legers (segmentation, estimation de poses) ; l'inference Pixal3D reste sur RunPod.
- Volume reseau RunPod en region `euro`, ~26 Go de poids en cache, facture meme pod eteint
  (~4 EUR/mois). Ne pas le supprimer : il evite de retelecharger les poids.
- Cle SSH locale : `~/.ssh/depthscan_sp005_ed25519` (pas de `id_ed25519`).
- Les logs GitHub Actions exigent une authentification meme sur un depot public ; le workflow
  publie la queue d'un build echoue en annotation de check-run, lisible anonymement.
