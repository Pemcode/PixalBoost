# GUI d'essais

> Contrat de sprint F08. Cette interface rend le pipeline actuel observable ; elle ne declare
> pas disponibles des capacites que le depot ne possede pas encore.

## Methode Harness Engineering

Cette tranche suit explicitement le cours
[Learn Harness Engineering](https://walkinglabs.github.io/learn-harness-engineering/en/) :

- F08 vit dans la source de verite machine avec comportement, commande de verification, etat et
  preuve, selon le principe des
  [feature lists](https://walkinglabs.github.io/learn-harness-engineering/en/lectures/lecture-08-why-feature-lists-are-harness-primitives/) ;
- les tests traversent la vraie fenetre, le controleur, un vrai processus enfant et le vrai builder
  PixaBoost, conformement a la
  [verification de bout en bout](https://walkinglabs.github.io/learn-harness-engineering/en/lectures/lecture-10-why-end-to-end-testing-changes-results/) ;
- phase, progression, commande sanitisee, journaux, duree, code retour et artefacts rendent le
  runtime observable, comme le demande la
  [lecon sur l'observabilite](https://walkinglabs.github.io/learn-harness-engineering/en/lectures/lecture-11-why-observability-belongs-inside-the-harness/).

Une revue maker-checker distincte a transforme chaque defaut important trouve (courses Qt,
annulation Windows, secret, artefact perime) en test de non-regression avant le passage de F08.

## Objectif utilisateur

Depuis une petite fenetre PyQt6, lancer un essai CPU ou mono-vue sans figer l'interface et voir
en direct :

- l'etat du processus et sa phase ;
- la progression lorsqu'elle est mesurable ;
- la commande courante, sans secret ;
- stdout/stderr, le temps ecoule et le code retour ;
- les artefacts locaux produits ou deja presents.

## Commandes du MVP

| Essai | Commande | Cout externe | Artefact attendu |
|---|---|---:|---|
| Gate CPU | `uv run poe check` | aucun | aucun |
| Tests CPU | `uv run poe test` | aucun | aucun |
| Benchmark synthetique | `uv run python -m pixaboost.bench.build --events-jsonl` | aucun | `data/bench/manifest.json` |
| Mono-vue Pod | `pixaboost reconstruct single-view ... --backend ssh-pod` | temps du Pod existant, seulement apres confirmation sur miss | `artifacts/<cle>/output.glb` + `runs/<id>/` |

Le benchmark emet des lignes `PIXABOOST_EVENT {...}`. Elles forment un petit contrat stable entre
le processus et la GUI. Les autres lignes restent visibles comme journal humain.

## Etats et fin observable

```text
idle -> starting -> running -> succeeded
                         |  -> failed
                         |  -> cancelling -> cancelled
```

Un code retour nul ne suffit pas lorsqu'un artefact est requis : le manifeste attendu doit avoir
ete cree ou modifie par la commande et contenir un objet JSON valide. Une sortie sans manifeste,
avec un ancien manifeste inchange ou avec un manifeste illisible est affichee comme un echec.

## Frontiere de dependances

```text
widgets PyQt6
    -> controleur QProcess -> commandes CPU publiques
    -> controleur QThread -> service mono-vue cache-first -> backend SSH Pod
```

- `core/` ne connait ni Qt, ni processus, ni interface.
- Le controleur accepte des specifications de commande injectees pour etre teste hors reseau.
- La commande affichee est sanitisee avant tout signal ou journal.
- L'annulation stoppe l'arbre du processus local sous Windows, pas seulement le wrapper `uv`.
- Le preflight mono-vue (taille, hash, cache) tourne hors du fil Qt et avant la confirmation.
- Un hit de cache ne construit pas le client SSH. Un miss exige une autorisation en memoire,
  ephemere et a usage unique avant toute connexion au Pod deja actif.
- Le backend refuse les hotes inconnus, borne les transferts et timeouts, verifie revisions,
  SHA-256 et en-tete GLB, et ne declare jamais un arret distant sans preuve.
- `artifacts/`, `data/bench/` et `runs/` sont seulement lus par l'inventaire ; aucun GLB lourd
  n'est charge dans le fil graphique.

## Exclusions explicites

- Pas de multi-vues : F10 n'est pas commencee et les poses des photos reelles sont inconnues.
- Pas de pourcentage Pixal3D invente : seules les etapes reellement emises sont affichees.
- Pas de provisionnement, demarrage ou suppression de Pod ; la cible doit deja etre active.
- Pas d'achat, de recharge ou d'activation de credits par PixaBoost.
- Si l'acquittement distant est inconnu, l'UI le dit et ne se ferme pas silencieusement ; le
  latch local interdit neanmoins toute nouvelle etape distante apres la demande d'annulation.
- Pas d'historique parallele : `runs/<ts>/` reste la source de verite des experiences.

## Verification F08

```powershell
uv run pytest -q tests/unit/test_gui_model.py tests/unit/test_gui_remote_adapter.py `
  tests/integration/test_gui.py tests/integration/test_gui_remote.py `
  tests/integration/test_gui_ux.py
```

La verification couvre le contrat d'evenements, la redaction des secrets, le single-flight, le
streaming d'un vrai processus enfant, les sorties succes/echec/annulation, l'arret des descendants,
la fraicheur et la validite de l'artefact obligatoire, le cache-first sans SSH, la confirmation
sur miss, le preflight non bloquant, les courses d'annulation/fermeture et un demarrage Qt
offscreen. Le gate global `uv run poe check` doit ensuite rester vert.
