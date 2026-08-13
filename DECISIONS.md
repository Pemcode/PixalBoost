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

## ADR-0008 — La stack est verrouillee sur torch 2.6 / cu124, donc sur sm_50 a sm_90

**Date** : 2026-08-12 · **Statut** : accepte · **Origine** : premier smoke test reel

**Constat.** Les extensions CUDA de Pixal3D (`o_voxel`, `flex_gemm`, `cumesh`, `nvdiffrast`,
`flash_attn_3`, `natten`) sont des wheels precompiles contre l'ABI de **torch 2.6.0 / CUDA 12.4**.
Aucun build de ces wheels n'existe pour une version superieure. Monter torch casserait les six
pour en reparer une.

**Consequence.** Le GPU doit avoir une compute capability comprise entre **5.0 et 9.0**.

| Capability | GPU | Verdict |
|---|---|---|
| sm_120 | RTX PRO Blackwell, RTX 5090, B200 | **incompatible** — torch 2.6 ne les connait pas |
| sm_90 | H100, H200 | compatible, cher |
| **sm_89** | **RTX 4090, RTX 4080, L40S, L4** | **compatible — meilleur rapport cout/perf** |
| sm_86 | A6000, A40, RTX 3090 | compatible |
| sm_80 | A100 | compatible |

Verifie a nos depens : une RTX PRO 4500 Blackwell (sm_120) a ete provisionnee et torch a refuse
de l'utiliser. Cette table existe pour que la question ne se represente pas.

**Reexamen.** Si un jour des wheels torch >= 2.7 apparaissent pour les six extensions.

---

## ADR-0009 — natten officiel plutot que le wheel amont

**Date** : 2026-08-12 · **Statut** : accepte · **Origine** : premier smoke test reel

**Constat.** Le wheel `natten 0.21.0` epingle par `requirements-hfdemo.txt` est un build prive
ne contenant que des noyaux **sm_90** — verifie en parsant les en-tetes ELF des cubins :
182 cubins, tous sm_90. Il fonctionne sur le H100 du Space Hugging Face et nulle part ailleurs.
Sur une RTX 4090 il provoque `no kernel image is available for execution on the device`, **en
cours d'inference**, apres le telechargement des poids.

**Decision.** Installer par-dessus le build **officiel SHI-Labs 0.17.5** pour la combinaison
exacte torch260/cu124/cp310, qui couvre `sm_60` a `sm_90`. Il expose l'ancienne API
`natten.functional`, que l'upsampler NAF gere deja : son import est enveloppe dans un
`try/except` qui selectionne `legacy_attention` quand la nouvelle API est absente.
`--no-deps` est obligatoire : pip ne doit pas toucher a torch.

**Alternatives rejetees.**
- *Compiler natten pour sm_89* : une a trois heures de CI, alors qu'un binaire officiel existe.
- *Monter torch a >= 2.7* pour activer le backend `flex-fna` sans noyaux : casse l'ABI de six
  extensions (voir ADR-0008).
- *Louer des H100* : le wheel amont y fonctionne, mais le cout horaire est prohibitif pour
  remplir un cache d'artefacts en batch.

**Verification.** `verify_extensions.py` lit desormais les architectures des cubins installes et
fait echouer le build si un GPU cible n'est pas servi. Confronte aux deux wheels, il rejette le
sm_90-only et accepte l'officiel.

---

## ADR-0010 — Detourage force sur BiRefNet MIT

**Date** : 2026-08-12 · **Statut** : accepte

**Constat.** `pipeline.json`, telecharge avec les poids, impose `briaai/RMBG-2.0` au module de
detourage — un depot **gated**, sous licence `other`. Le defaut ecrit dans le code source
(`BiRefNet.py:9`) est `ZhengPeng7/BiRefNet`, mais il est ecrase par la configuration.

**Decision.** Forcer `ZhengPeng7/BiRefNet` (MIT, non gated). RMBG-2.0 **est** l'architecture
BiRefNet reentrainee par BRIA ; la classe amont fonctionne avec l'un comme avec l'autre, seul le
nom du depot change.

**Raison.** Une licence `other` avec validation d'acces dans une chaine destinee a devenir une
application est un risque juridique gratuit, alors qu'un equivalent MIT existe.

**Portee.** Le detourage est **separable** du reste : le benchmark synthetique rend sur fond noir
avec masques exacts, donc F12 ne l'utilise pas du tout. Il ne concerne que les photos reelles, et
son remplacement est contenu dans `backends/`.

---

## ADR-0006 — L'ICP raffine, il ne cherche pas : bassin mesure a ~20 degres

**Date** : 2026-08-07 · **Statut** : accepte · **Origine** : F11

**Mesure.** Sur un nuage non structure, l'ICP point-a-point converge exactement (rmse ~5e-17)
tant que l'erreur de rotation initiale reste sous **environ 20 degres**. Au-dela il se fige dans
un minimum local. Mesure balayee offset par offset ; le test
`test_icp_converges_from_anywhere_inside_the_basin` verrouille le constat.

**Decision 1.** L'espacement des redemarrages doit rester **inferieur au bassin**. Six
redemarrages a 60 degres laissent des orientations litteralement inatteignables, et le mauvais
ajustement qui en resulte se lit ensuite comme une confiance faible alors que c'est la recherche
qui a echoue. Defaut porte a **24 redemarrages** (15 degres).

**Decision 2.** Un **pre-alignement grossier** (centroides et etendue) precede toujours l'ICP.
Sans lui, l'ICP doit absorber simultanement une translation, un facteur d'echelle et une rotation,
et echoue meme a quelques degres.

**Consequence pour F10.** L'ICP seul ne recupere pas une pose depuis rien. Deux voies restent
ouvertes, a trancher au moment de F10 : partir des **poses nominales de la prise de vue** (6
azimuts x 3 elevations, connues a quelques degres pres — c'est le chemin privilegie), ou ajouter
un recalage global par descripteurs. On ne construit pas le second tant que le premier n'a pas
ete essaye.

**Alternative rejetee.** *Augmenter indefiniment le nombre de redemarrages* : le cout croit
lineairement et ne resout pas le cas ou la rotation relative n'est pas autour de l'axe vertical
(deux vues d'elevations differentes). Les poses nominales le resolvent directement.

---

## ADR-0007 — La confiance de recalage mesure l'ambiguite, pas seulement le residu

**Date** : 2026-08-07 · **Statut** : accepte · **Origine** : F11

**Decision.** `confidence = inlier_ratio x qualite_d_ajustement x distinctness`, ou *distinctness*
compare le meilleur ajustement au meilleur ajustement **materiellement different** (plus de 10
degres d'ecart). Sous le seuil, `register` **leve une exception** au lieu de renvoyer une pose.

**Raison.** Le projet vise des **pieces mecaniques**, qui sont couramment symetriques. Mesure sur
une bague a symetrie d'ordre 12 : elle s'aligne sur elle-meme **exactement** — rmse 1,3e-16, 100 %
d'inliers. Un score fonde sur le seul residu lui donnerait une confiance de 1,0 et renverrait une
pose choisie au hasard parmi douze. La meme mesure sur une piece asymetrique donne une
distinctness de 1,0 et une confiance de 1,0.

Autrement dit : **la qualite d'ajustement ne distingue pas « c'est la bonne pose » de « c'est
l'une des douze »**. Seul le terme d'ambiguite le fait.

**Pourquoi refuser plutot que signaler.** Une pose fausse ne degrade pas legerement la fusion
multi-vues : elle etale la piece, et sans rien lever. Un appelant qui veut inspecter un recalage
rejete passe `min_confidence=0.0` ; l'ambiguite reste rapportee, jamais masquee.

**Consequence.** Au moins deux initialisations sont exigees : avec une seule il n'existe aucune
rivale, donc une piece symetrique passerait le controle sans etre detectee. `register` leve une
`ValueError` dans ce cas plutot que de rendre un score d'ambiguite qui ne veut rien dire.

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

---

## ADR-0011 — La GUI est un harness d'essais, pas un second pipeline

**Date** : 2026-08-13 · **Statut** : accepte · **Origine** : F08

**Decision.** L'interface PyQt6 pilote les commandes CPU reproductibles dans `QProcess` et le
service mono-vue cache-first dans des workers `QThread`. Elle expose des evenements structures —
etat, phase, progression disponible, commande sanitisee, journal, duree, code retour et artefacts
— sans placer Qt dans `core/` ni implementer le protocole SSH dans la fenetre.

**F09** — et non F07 — fournit le transport vers un Pod **deja actif**. Le preflight local est
execute hors du fil graphique : un hit ne construit aucun client SSH ; un miss exige une
confirmation humaine explicite, ephemere et a usage unique. PixaBoost ne provisionne, ne demarre,
n'achete, ne recharge et n'active jamais de Pod ou de credits.

F07 reste `blocked` : il porte la preuve **materielle** (un GLB revenu d'un vrai GPU via la
commande de verification), que le contrat CPU de F09 ne remplace pas.

**Raison.** Le transport SSH strict verifie la revision du modele, borne et hache les transferts,
persiste le manifeste et relaie uniquement la telemetrie disponible. Son annulation est prudente :
aucune nouvelle etape ne part apres la demande, mais l'UI affiche `unknown` tant qu'un arret du
worker distant n'est pas acquitte. Aucun faux pourcentage ni faux succes d'annulation n'est cree.

**Frontiere.** `gui/` ne decide rien sur la reconstruction. Un controleur de processus traduit
les sorties de commandes en evenements types ; la fenetre ne fait qu'afficher ces evenements.
Les preuves experimentales restent dans `runs/<ts>/`. L'inventaire de fichiers locaux est une
aide de navigation, jamais une seconde source de verite ni une validation scientifique.

**Alternatives rejetees.**
- *Copier directement la GUI miniDS* : son worker est couple a son client HTTP et sa fenetre
  melange trop de responsabilites ; PixaBoost n'a pas le meme contrat distant.
- *Appeler le benchmark dans le fil Qt* : bloque la fenetre et supprime l'observabilite du
  cycle de vie. `QProcess` fournit une isolation et une annulation locale explicites.
- *Ajouter du SSH directement dans la fenetre* : viole la frontiere backend ; le runner injecte
  appelle a la place le service public puis `backends/ssh_pod.py`.
- *Persister un historique GUI separe* : contredirait `runs/<ts>/`, deja designe comme source de
  verite des experiences.

**Reexamen.** Avant une mise en production serverless : le transport de recherche actuel cible
volontairement un Pod existant et ne gere aucun cycle de vie RunPod.

---

## ADR-0012 — `trials/` est la couche d'orchestration, entre `backends/` et les interfaces

**Date** : 2026-08-13 · **Statut** : accepte · **Origine** : F09

**Decision.** Un quatrieme paquet, `src/pixaboost/trials/`, porte l'orchestration d'une tentative
de reconstruction : resoudre le cache, exiger l'autorisation, appeler l'adaptateur, ecrire
`runs/<id>/manifest.json`, `metrics.json` et `logs.jsonl`. Il est appele **a la fois** par la CLI
(`pixaboost reconstruct single-view`) et par la GUI. La carte du depot dans `CLAUDE.md` est
amendee en consequence.

**Raison.** Cette logique ne pouvait vivre nulle part ailleurs sans casser une contrainte :

| Emplacement | Pourquoi c'est refuse |
|---|---|
| `core/` | Ecrit des fichiers, lit une image, depend de `backends/`. Contraintes n°4 et n°5. |
| `backends/` | La contrainte n°5 y interdit toute logique metier ; l'ordre cache -> autorisation -> inference -> preuve en est. |
| `gui/` | La CLI en a besoin aussi. Dupliquer, c'est garantir la divergence. |

Le critere de decoupe est donc **la contrainte n°7** : la production du manifeste est ce qui rend
une experience recevable, et elle doit etre commune a toutes les interfaces.

**Ce que `trials/` n'a pas le droit de faire.** Aucune decision sur la reconstruction elle-meme :
ni fusion, ni ponderation, ni score de confiance geometrique. Il sequence et il prouve. La
contrainte n°11 reste entiere.

**Alternatives rejetees.**
- *Mettre l'orchestration dans `cli.py` et la reappeler depuis la GUI* : la CLI deviendrait une
  dependance de la GUI, et le manifeste dependrait du parsing d'arguments.
- *La mettre dans `gui/single_view_adapter.py` seulement* : rend la CLI impossible a tester sans
  Qt, donc le gate CPU dependrait de PyQt6.

**Reexamen.** Si une troisieme interface apparait, ou au moment de F10 : le multi-vues ajoutera
une seconde orchestration, et c'est la que se verra si l'abstraction tient.

---

## ADR-0013 — Le cache reserve la cle avant de facturer, et refuse un artefact altere

**Date** : 2026-08-13 · **Statut** : accepte · **Origine** : F09

**Decision.** `ArtifactCache` acquiert une **reservation interprocessus** sur une cle de contenu
avant de lancer un travail facture, et `load()` **leve** `CacheCorruptionError` au lieu de
renvoyer `None` quand une entree existe mais ne correspond plus a son SHA-256.

**Raison.** Les deux comportements protegent la meme chose — la contrainte n°9 — contre deux
defaillances opposees.

1. *Sans reservation*, deux processus qui demandent la meme reconstruction paient deux fois : la
   verification du cache et l'appel GPU n'etaient pas atomiques. La primitive portable est la
   creation de repertoire (`mkdir` echoue si le repertoire existe), la seule qui soit atomique a
   la fois sur NTFS et sur ext4. Un proprietaire vivant n'est jamais vole ; un worker tue est
   reclame via son PID et son hostname.
2. *Sans verification d'integrite*, un GLB tronque se lit comme un **hit**. Le pire scenario n'est
   pas le plantage, c'est la metrique F12 calculee sur un artefact corrompu et archivee comme
   resultat. Un miss silencieux serait presque aussi grave : il relancerait un travail facture
   sans jamais signaler que le cache est abime.

**Pourquoi lever plutot que reparer.** Regenerer automatiquement une entree corrompue depense de
l'argent sur un evenement qui ne devrait pas se produire. L'humain doit voir la corruption.

**Verification.** Prouve par injection, pas par construction : `test_cache.py` altere de vrais
octets stockes, tue un vrai processus detenteur, et verifie qu'un proprietaire vivant resiste au
vol.

---

## ADR-0014 — Accepter une licence gated pour SAM 3, contre le precedent d'ADR-0010

**Date** : 2026-08-13 · **Statut** : accepte · **Origine** : F14

**Decision.** Le projet utilise `facebook/sam3` pour la segmentation par clic, alors que ce depot
est **gated** et sous `license: other` (« SAM License » de Meta). C'est exactement la propriete
pour laquelle ADR-0010 a rejete `briaai/RMBG-2.0`.

**Ce que la verification a etabli** (aout 2026, sources primaires, pas de memoire) :

| | SAM 3 | SAM 2.1 Hiera-Large |
|---|---|---|
| Licence | « SAM License » custom, **gated** | **apache-2.0**, non gated |
| Parametres | 0,9 Md | 0,2 Md |
| VRAM inference | < 4 Go | nettement moins |
| Clic (points) | oui, via `Sam3TrackerModel` | oui |
| Usage commercial | autorise, avec restrictions | sans restriction |

Restrictions de la SAM License : conformite aux controles a l'export (pas d'usage militaire,
nucleaire, armement), pas de retro-ingenierie, attribution en publication, redistribution des
derives sous la meme licence. Pas de seuil d'utilisateurs actifs.

**L'argument contre, enonce en toute franchise.** La nouveaute de SAM 3 est la segmentation par
*concept* (texte, exemplar). Notre besoin est un **clic**, c'est-a-dire de la segmentation
visuelle promptable — ce que SAM 1 et SAM 2 font depuis toujours, et ce que SAM 3 lui-meme
delegue a `Sam3TrackerModel`, decrit en amont comme « SAM 2 avec la meme API ». Autrement dit :
**on paie un cout juridique et operationnel pour une capacite qu'on n'utilise pas.** ADR-0010
avait rejete RMBG-2.0 sur ce raisonnement precis, avec un equivalent MIT disponible.

**Pourquoi la decision est prise quand meme.** Le proprietaire du depot a tranche apres que ce
conflit lui a ete presente explicitement, jeton Hugging Face fourni a l'appui. C'est une decision
de produit qui lui revient, et l'ecart de qualite de masque sur le cas difficile — outillage
metallique au contact d'une piece metallique, meme matiere et memes reflets — n'est pas mesure et
pourrait justifier l'ecart.

**Ce que la decision engage concretement.**
1. Un jeton circule desormais dans le code. `backends/sam3.py` ne le fait jamais apparaitre dans
   une exception, et un test injecte une vraie fuite (`401 ... Bearer hf_...`) pour le prouver.
2. Tout nouveau contributeur, et tout pod reconstruit, doit accepter la licence sur le Hub avec le
   compte proprietaire du jeton. C'est un point de friction manuel, comme `setup_pod.sh`.
3. L'extra `segmentation` reste hors du groupe `dev` : ~3 Go, et le gate CPU doit rester
   installable en CI gratuite.

**Condition de reexamen — ecrite pour etre falsifiable.** Si, sur les photos reelles, SAM 2.1
(Apache) produit des masques de qualite comparable a SAM 3 sur le cas « pince au contact de la
piece », le cout juridique n'est plus paye pour rien et **on repasse a SAM 2.1**. La mesure
n'a pas ete faite. Tant qu'elle ne l'est pas, ADR-0014 repose sur une preference, pas sur une
mesure — et le present paragraphe existe pour que ce fait ne se perde pas.

**Frontiere preservee.** `core/` ne connait ni SAM, ni torch, ni le Hub : il ne manipule que des
masques booleens et des tableaux uint8. La contrainte n°4 est intacte, et la contrainte n°11
aussi — rien ici ne fusionne quoi que ce soit.
