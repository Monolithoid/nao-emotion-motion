# nao-emotion-motion

Apprentissage et génération de mouvements expressifs des **bras** et de la **tête** de NAO, à partir de démonstrations physiques, conditionnés par une **émotion** et une **intensité** (0 à 1).

Le composant moteur n'est pas un LLM. C'est un petit modèle neuronal supervisé (PyTorch) dont l'inférence est légère et qui peut aussi tourner en numpy seul. Un LLM extérieur peut fournir l'émotion et l'intensité, mais n'obtient qu'une **proposition** de mouvement : l'exécution reste une action explicite de l'opérateur.

> **État : premier socle, validé en simulation uniquement.** Les données synthétiques servent à vérifier la chaîne technique. Elles ne disent rien de l'expressivité. Rien n'a encore été testé sur un robot réel.

---

## Démarrage rapide (sans robot)

```bash
pip install -e ".[dev]"          # numpy, pyyaml, torch (+ pytest)
nao-motion check-config          # vérifie configs/ et liste ce qui manque pour le robot réel
nao-motion demo                  # parcours complet en simulation (~15 s), dans ./demo_sim
cp .env.example .env             # identité et adresse du robot (lue par la config et par le pont)
pytest                           # tests
```

`nao-motion demo` déroule les étapes du premier résultat :

0. procédure de calibration exercée sur le robot simulé (sans portée réelle, non activée) ;
1. collecte et sauvegarde de démonstrations des bras (démonstrateur synthétique, adaptateur simulé) ;
2. consultation d'une prise et relecture virtuelle validée par le guardian ;
3. entraînement du modèle des bras ;
4. génération, validation, puis exécution du cycle complet sur le robot simulé ;
5. collecte tête : le **modèle des bras** joue chaque émotion/intensité pendant que la tête est guidée (ici par un démonstrateur simulé), avec plusieurs contextes de regard par geste ;
6. entraînement de la tête avec le modèle des bras figé, puis exécution simulée avec un événement.

## Interface locale de collecte

```bash
nao-motion ui                    # http://127.0.0.1:8765 (adaptateur simulé par défaut)
nao-motion ui --adapter bridge   # vers le robot via le pont NAOqi (voir plus bas)
```

L'interface écoute uniquement sur `127.0.0.1`. Elle a six onglets :

- **1 · Bras** : choisir l'émotion, l'intensité et éventuellement une intention. La consigne s'affiche en grand (« surprise — 0,8 »). Ensuite : *Préparer* (relâche les bras et maintient la tête), *Démarrer* (compte à rebours, puis enregistrement des angles **mesurés** et de leurs horodatages), *Arrêter*. Un aperçu des courbes s'affiche sans aucun mouvement, puis on choisit *Accepter*, *Garder en attente*, *Rejeter* ou *Recommencer*, avec une annotation possible.
- **2 · Tête** : choisir la source du mouvement des bras. Par défaut, c'est un **geste généré par le modèle des bras** (émotion, intensité, variation, graine) ; une démonstration enregistrée peut aussi être rejouée. Définir ensuite le contexte événementiel (aucun événement, événement de test, ou dernier événement reçu du composant externe). *Préparer* valide le geste, fait une approche motorisée des bras vers son début, puis relâche la tête. *Démarrer* exécute les bras pendant que vous guidez la tête. On enregistre émotion/intensité, bras **réellement mesurés**, consigne envoyée et tête. Plusieurs contextes de regard peuvent être enregistrés pour un même geste (même graine).
- **Prises** : répartition par émotion × intensité (acceptées / total), liste filtrable, courbes, relecture virtuelle (curseur animé, relecture lissée superposée), statut et annotations.
- **Entraînement** : lancer l'entraînement des bras ou de la tête et consulter les résultats (RMSE en entraînement et en validation, référence de comparaison, taux d'acceptation du guardian).
- **Génération** : choisir une émotion, éventuellement un mélange de deux émotions, l'intensité, une variation et l'événement. On obtient les courbes, le rapport du guardian sur le geste et sur le cycle complet, et un bouton *Exécuter* (désactivé si le cycle est refusé).
- **Robot** : *Connecter* (sans mouvement), *Acquitter*, *Raidir sur place*, *Retour au repos* (explicite), *Relâcher*.

Le bouton **ARRÊT** interrompt l'enregistrement, la relecture et l'exécution. La posture est maintenue et le contrôleur est désarmé. Il n'y a **pas** de retour automatique au repos.

En simulation, un démonstrateur synthétique remplace votre main. Les prises qu'il produit sont marquées `synthetic: true`.

### Points d'entrée externes (HTTP JSON)

| Route | Rôle |
|---|---|
| `POST /api/event` | Événement du composant caméra/micros : `{"present": true, "azimuth": 30, "elevation": 5, "unit": "deg", "frame": "torso" \| "head", "source": "camera"}`. Avec `frame: "head"`, joindre `head_yaw_rad` et `head_pitch_rad` au moment de la mesure, sinon ce sont les angles lus à la réception qui sont utilisés. `{"present": false}` signale l'absence d'événement. `ttl_s` est optionnel (au-delà, l'événement est considéré comme absent). |
| `POST /api/request` | Pour un LLM : `{"emotion": "joie", "intensity": 0.7}` ou `{"mix": {"joie": 0.5, "surprise": 0.5}, ...}`. La réponse contient la trajectoire, le rapport du guardian et un `plan_id`. **Rien n'est exécuté.** |
| `POST /api/execute` | `{"plan_id": ..., "confirm": true}`, avec contrôleur armé : action de l'opérateur. |

## Ligne de commande

```bash
nao-motion synth                        # remplit data/ de prises SYNTHÉTIQUES (tests techniques)
nao-motion takes --stats                # liste et répartition des prises
nao-motion train arms                   # entraîne les bras (prises « accepted »)
nao-motion train head                   # entraîne la tête, modèle des bras figé (dernier run)
nao-motion generate --emotion joie --intensity 0.7 --event-az 30 --out geste.json
nao-motion generate --mix '{"joie":0.6,"calme":0.4}' --intensity 0.5 --variation 0.03 --seed 1
nao-motion generate --emotion peur --intensity 0.9 --simulate     # exécution sur le robot SIMULÉ
nao-motion execute geste.json --adapter bridge --confirm          # seul chemin CLI vers un robot réel
nao-motion robot-info                   # identité, NAOqi, limites d'usine, posture mesurée (lecture seule, via le pont)
nao-motion calibrate                    # procédure de calibration du guardian (voir « Calibration »)
nao-motion calibrate --adapter sim      # la même procédure sur le robot simulé
nao-motion calibration-activate configs/calibration/<fichier>.yaml
nao-motion capture-pose --group left_arm --name start --adapter bridge   # lit les capteurs, n'envoie rien
nao-motion serve-inference --port 8766  # inférence déportée (génère, n'exécute pas)
```

---

## Architecture

```
configs/                YAML versionnés : articulations + limites, poses, émotions, guardian, robot, modèle
bridge/naoqi_bridge.py  pont NAOqi (Python 2.7 ou 3), seul fichier dépendant du SDK
nao_motion/
  config/               chargement et validation des configurations
  events.py             contexte événementiel, repères, conversions
  trajectory.py         trajectoire horodatée (s, rad)
  data/                 stockage des prises, représentation (B-spline), rééchantillonnage
  models/               features, MLP PyTorch, checkpoints, export numpy, jeux d'apprentissage
  training.py           entraînement bras / tête, métriques, découpage par groupes
  engine.py             génération bras puis tête (propose, n'exécute pas)
  guardian/             validation indépendante du trajet complet
  control/              contrôleur (cycle, exécution, arrêt), enregistrement en deux passages
  adapters/             interface commune, simulation, client du pont NAOqi
  inference/            serveur et client d'inférence déportée
  ui/                   interface web locale (bibliothèque standard + HTML/JS)
  workbench.py          couche de service commune à l'UI, la CLI et la démo
  sim/synthetic.py      démonstrateurs synthétiques (simulation uniquement)
```

Les données vont dans `data/`, les entraînements dans `runs/`. Ces deux dossiers sont ignorés par git.

### Deux modèles, un moteur

| | Entrées | Sortie |
|---|---|---|
| **Bras** | émotion (one-hot, ou mélange pondéré), intensité | 8 points de contrôle × 10 articulations + log(durée) |
| **Tête** | émotion, intensité, **points de contrôle des bras prévus** + leur durée, orientation actuelle de la tête, événement `[présent, sin az, cos az, élévation]` | 8 points de contrôle × 2 articulations |

La tête est exprimée sur la **même phase temporelle** que les bras, avec la même durée. Le réseau voit donc tout le déroulement prévu des bras et peut apprendre une anticipation ou un retard. Rien ne force la tête à suivre les mains : ce que montrent les démonstrations (regarder les mains, rester vers l'interlocuteur, détourner le regard) détermine son comportement.

**Collecte et entraînement de la tête.** Le modèle des bras déjà entraîné reçoit une émotion et une intensité. Le geste qu'il produit est validé par le guardian, puis exécuté par les bras pendant que l'opérateur guide uniquement la tête. Chaque prise donne un exemple (émotion/intensité, bras réalisés, tête enregistrée, événement). On peut aussi rejouer une démonstration de bras enregistrée.

« Bras figés » veut dire que les poids du modèle des bras ne changent pas pendant l'entraînement de la tête. Il est chargé sans gradient, et l'empreinte de son checkpoint est enregistrée dans celui de la tête et dans chaque prise de tête générée. Par défaut (`head_model.arm_input: recorded`), l'entrée « bras » d'un exemple est la trajectoire des bras **mesurée** pendant la prise, c'est-à-dire le mouvement réellement exécuté. L'option `predicted` utilise plutôt la sortie du modèle des bras.

### Représentation des mouvements

- Les prises **brutes** (angles mesurés, horodatages réels) ne sont jamais modifiées.
- Chaque prise a un **segment utile**. Pour une prise de bras, il est proposé automatiquement en retirant l'immobilité du début et de la fin, et il reste modifiable. Pour une prise de tête, c'est la fenêtre exacte de relecture des bras.
- Le segment est ramené à une **phase** u ∈ [0, 1] par rééchantillonnage linéaire sur 100 points, puis projeté sur **8 points de contrôle B-spline cubique** par articulation. La **durée** est apprise à part, en log.

Pourquoi ce choix : avec peu de données, il faut peu de paramètres de sortie. Les courbes obtenues sont lisses (vitesse et accélération continues, ce qui aide le guardian). La relation entre paramètres et trajectoire est linéaire, donc interpoler entre exemples ou entre émotions reste cohérent. Cet alignement linéaire ne recale pas les sous-phases d'un geste (pas de DTW) ; c'est une évolution possible.

### Moyenne des gestes et variantes

Le premier modèle est une **régression déterministe**. Si « joie » a été démontrée de deux façons différentes, il apprend la **moyenne** des deux, qui peut être un geste jamais montré. Le démonstrateur synthétique contient volontairement un tel cas. Pour le premier prototype, les variantes sont obtenues ainsi :

- **interpolation** : intensité continue et mélange pondéré d'émotions (`--mix`) ;
- **variation** : un bruit lisse et reproductible (`--variation`, `--seed`) sur les points de contrôle intérieurs, non appris.

Évolution prévue sans changer l'interface : les MLP acceptent déjà une entrée latente `latent_dim` (0 par défaut). Elle permettra un CVAE ou un mélange de prototypes, qui produiront des variantes **apprises** à partir de démonstrations multiples.

### Mesurer plutôt qu'imposer un nombre de prises

Chaque entraînement écrit dans `runs/<run>/metrics.json` :

- la RMSE (rad) sur trajectoires denses en entraînement et en validation, au total et par articulation ;
- une **référence** de comparaison : la moyenne des prises de même émotion et d'intensité voisine. Si le modèle ne fait pas mieux, il faut le voir ;
- l'erreur de durée ;
- le taux de trajectoires générées acceptées par le guardian sur une grille émotion × intensité (× événement pour la tête) ;
- un avertissement quand les données ne permettent pas de validation (moins de deux groupes).

Le découpage entraînement/validation se fait **par groupe** : une prise de bras et toutes les prises de tête enregistrées sur elle restent du même côté. Aucune variante d'une même prise ne peut fuiter.

---

## Unités, axes et repères

- **Angles** : radians. **Temps** : secondes. Noms d'articulations : ceux de NAOqi.
- **Repère commun des événements** : `FRAME_TORSO` de NAOqi. x pointe vers l'avant, y vers la gauche du robot, z vers le haut.
  - azimut = atan2(y, x), **positif vers la gauche** ;
  - élévation = atan2(z, √(x²+y²)), **positive vers le haut**.
- **Absence d'événement** : `present: false`, codée `[0, 0, 0, 0]`. Un azimut de 0 veut dire « devant » et se code `[1, 0, 1, 0]`. Les deux cas ne peuvent pas être confondus.
- **Conversion depuis la caméra (repère tête)** : rotation `Rz(HeadYaw) · Ry(HeadPitch)` avec les angles de tête au moment de la mesure. La translation du cou est négligée, ce qui reste raisonnable au-delà d'environ 50 cm.
- **Conventions NAOqi** : HeadYaw > 0 tourne la tête à gauche, HeadPitch > 0 la baisse. Regarder une élévation *e* correspond donc à peu près à HeadPitch = −*e*.
- Les événements sont toujours stockés dans le repère commun, avec leur source et leur horodatage.

## Format des données

Chaque prise est stockée dans `data/takes/<take_id>/` :

- `samples.npz` : `t` (N,) en s depuis le début de l'enregistrement, `q` (N, J) en rad **mesurés**, `joints` ; pour une prise de tête, aussi la consigne envoyée aux bras (`ref_t`, `ref_q`, `ref_joints`) sur la même horloge ;
- `meta.json` : `take_id`, `session_id`, `kind` (`arms` | `head`), `emotion`, `intensity`, `intention` (optionnelle), `joints`, `unit`, `time_unit`, `guided_joints`, `driven_joints`, `event` (repère torse), `event_frame`, `segment`, `head_initial`, `arm_source` (`take` | `generated`), `arm_take_id` (démonstration rejouée), `arm_generation` (émotion, intensité, variation, graine, empreinte du modèle des bras), `arm_gesture_id` (groupe de découpage), `calibration_version`, `config_fingerprint`, `adapter`, `sample_rate_hz`, `synthetic`, `status` (`pending` | `accepted` | `rejected`), `annotations` (`notes`, `tags`), `created_at` et `format_version`.

Les données sont refusées à l'enregistrement si une valeur n'est pas finie, si les horodatages ne sont pas strictement croissants, si l'émotion est inconnue, si l'intensité sort de [0, 1], si les unités ne sont pas celles attendues, si une articulation est à la fois guidée et pilotée, ou si une prise de tête n'a pas de prise de bras valide.

Les checkpoints (`runs/<run>/model.pt`) contiennent l'architecture, les poids, les articulations, la liste d'émotions, la normalisation, la représentation, la version de calibration, l'empreinte de configuration, les prises d'entraînement et de validation, les hyperparamètres et les métriques. Pour la tête, ils contiennent aussi la référence et l'empreinte du modèle des bras. Au rechargement, un modèle est **refusé** si ses émotions ou ses articulations diffèrent de la configuration courante, et un avertissement est émis si la calibration a changé. `model_numpy.npz` permet une inférence sans PyTorch.

L'emplacement de l'inférence se règle dans `configs/robot.yaml` → `inference` : `location: local | remote`, `backend: torch | numpy`, `remote_url`. Une trajectoire calculée à distance est toujours revalidée localement par le guardian.

---

## Guardian et exécution

Le modèle **propose**, le contrôleur **encadre**.

- **Limites du guardian** : elles viennent uniquement d'une **calibration** mesurée sur le robot (section suivante). Les valeurs de `joints.yaml` et `poses.yaml` ne servent qu'à la simulation.
- **Guardian** (`nao_motion/guardian`) : il accepte ou refuse, sans jamais corriger. Il refuse les formats invalides, les valeurs non finies, les articulations inconnues ou celles des jambes, les jeux d'articulations incompatibles et les durées hors bornes. Il vérifie les **limites min/max avec marge** et les **contraintes couplées** linéaires (`guardian.yaml`, y compris tête ↔ bras) sur le **trajet complet** échantillonné à 100 Hz, et pas seulement aux extrémités. Il limite la **vitesse** et l'**accélération** par articulation et contrôle l'écart entre le premier point et l'état mesuré. En cas de dépassement de vitesse, il indique le ralentissement qui suffirait, sans l'appliquer.
- Des min/max indépendants et quelques contraintes linéaires **ne garantissent pas l'absence de collision**. Les contraintes fournies sont des exemples, à remplacer par des zones mesurées sur votre robot.
- **Cycle nominal** construit par le contrôleur, non appris : repos → départ → geste → départ → repos. Les transitions repos ↔ départ ont un jerk minimal. Les transitions départ ↔ geste sont quintiques et raccordées en vitesse et en accélération. Le **cycle entier** est validé avant le premier envoi, puis revalidé contre l'état mesuré juste avant l'exécution. Il est refusé si le robot n'est pas mesuré au repos (sauf choix explicite `from_pose="current"`).
- **Exécution en streaming** (setAngles à 25 Hz). À chaque pas, le contrôleur vérifie l'arrêt demandé, la connexion (battement) et l'écart de suivi entre consigne et mesure.
- **Rôles articulaires** : une articulation est *guidée* (raideur nulle, uniquement lue), *pilotée* (raide, commandée) ou laissée telle quelle. Une commande vers une articulation guidée est refusée avant d'atteindre le robot : on ne suppose jamais qu'un axe relâché suit une consigne.
- **Arrêt et erreurs** :
  - *ARRÊT* : l'exécution s'interrompt, la posture **mesurée** est maintenue et le contrôleur est désarmé.
  - *Erreur pendant l'exécution* : le contrôleur tente de maintenir la posture, passe en `FAULT` et attend que l'opérateur acquitte.
  - *Perte de connexion* : plus rien ne peut être envoyé depuis le PC. Le **chien de garde du pont**, côté robot, arrête alors les tâches et maintient la posture si aucun battement n'arrive pendant 0,6 s ou si le client se déconnecte.
  - Il n'y a **jamais** de retour motorisé automatique au repos ni de relâchement automatique. *Retour au repos* et *Relâcher* sont des actions explicites, parce que la bonne réaction dépend de l'état réel du robot.
- **Aucun mouvement automatique** au démarrage, pendant les tests ou à la fin d'un entraînement. Le contrôleur démarre désarmé, et chaque mouvement exige l'armement puis une action explicite. Plusieurs tests le vérifient.
- **Robot réel** : l'armement est refusé sans calibration active, ou si le robot connecté ne porte pas le nom de la calibration ou du `.env`.
- **Redémarrages observés pendant certains mouvements** : leur cause n'est pas établie ici. Le guardian **ne les corrige pas** et ne doit pas être présenté comme une protection contre eux. Pour les étudier, il faudrait journaliser les mouvements concernés (trajectoire, vitesses, températures et courant si disponibles) et comparer.

---

## Calibration du guardian

La calibration est une procédure à part entière, faite robot par robot. Son résultat est un fichier versionné, `configs/calibration/<robot>-<date>.yaml`. Le fichier actif est désigné par `configs/calibration/ACTIVE`, et une calibration n'est jamais écrasée.

```bash
nao-motion calibrate            # interactive, via le pont ; rien n'est commandé
```

1. **Identification** : nom du robot, version de NAOqi, configuration matérielle, et limites d'usine lues par `ALMotion.getLimits`.
2. **Exploration guidée**, groupe par groupe (bras gauche, bras droit, tête). Le groupe exploré est relâché et seulement lu, les autres sont raidis sur place. Amenez lentement chaque articulation jusqu'à ses butées *utiles*, et parcourez les combinaisons (tête tournée **et** baissée, épaule levée **et** écartée, etc.).
3. **Poses** de repos et de départ, capturées pour chaque groupe en plaçant le robot à la main.
4. **Calcul** : les limites sont l'amplitude mesurée, intersectée avec les limites d'usine, moins une marge (`configs/calibration.yaml`). Les **contraintes couplées** viennent de l'enveloppe convexe de chaque couple exploré (`pairs`), simplifiée de façon conservatrice. La procédure est refusée si l'exploration est trop courte ou trop étroite, ou si une pose sort de l'enveloppe.
5. **Relecture** du résumé, puis enregistrement et activation (`nao-motion calibration-activate`).

Une calibration active remplace limites, poses et version de calibration dans toute la chaîne (guardian, prises, checkpoints). Les contraintes manuelles de `guardian.yaml` s'y ajoutent, notamment entre membres (bras ↔ tête, bras ↔ bras), qu'une exploration articulation par articulation ne mesure pas. L'enveloppe convexe suppose une zone atteignable convexe pour chaque couple ; une zone en L doit être découpée à la main.

## Connecter le robot réel

Identité relevée sur le NAO de Joachim (dans `.env.example`) : **naoH25, tête/corps/bras VERSION_40 (NAO V4), NAOqi 2.1.4.13, Python 2.7.3, nom « nao »**. Pour la relever ou la vérifier sur un autre robot, en SSH (`ssh nao@<IP>`) :

```bash
qicli call ALSystem.systemVersion        # version NAOqi
qicli call ALSystem.robotName            # nom (lié à la calibration)
qicli call ALMotion.getRobotConfig       # modèle et versions tête/corps/bras
qicli call ALMotion.getLimits HeadPitch  # [min, max, vitesse max, couple max], par articulation
python --version                         # Python embarqué
```

Étapes :

1. `cp .env.example .env` (non versionné) : identité attendue du robot, port du pont. Ces valeurs remplacent celles de `configs/robot.yaml`.
2. **Installer le SDK Python NAOqi 2.1.4** (Python 2.7) sur la machine qui fera tourner le pont, ou lancer le pont directement sur le robot, qui a Python 2.7.3. Le pont essaie `import qi`, puis `from naoqi import ALProxy`.
3. **Préparer la position du robot.** Les jambes restent hors périmètre et ne sont jamais commandées. Placez le robot dans une posture stable, par exemple assis ou accroupi. Gérez la vie autonome, la respiration et les mouvements d'attente depuis vos outils habituels : le pont n'y touche pas, car les désactiver peut faire bouger le robot selon les versions.
4. **Lancer le pont**, qui n'envoie aucun mouvement au démarrage et attend l'adresse du robot :
   ```bash
   python2 bridge/naoqi_bridge.py            # adresse fournie ensuite par l'interface
   python2 bridge/naoqi_bridge.py --robot-ip nao.local   # ou dès le lancement
   ```
   Pour tester le protocole sans robot : `python bridge/naoqi_bridge.py --fake`. Le pont n'accepte qu'un client à la fois (interface **ou** commande).
   Puis, dans `nao-motion ui --adapter bridge` → onglet **Robot**, saisir l'**adresse du robot (IP ou nom DNS)**, par exemple `192.168.1.10` ou `nao.local`, et cliquer sur *Connecter au robot*. Cette action n'envoie aucun mouvement et ne change aucune raideur. L'adresse est mémorisée dans le navigateur. En ligne de commande, `--robot <IP ou DNS>` joue le même rôle, avec `NAO_IP` comme valeur par défaut.
5. `nao-motion robot-info --robot <IP ou DNS>` : vérifie la connexion et affiche identité, limites d'usine et posture mesurée, sans aucun mouvement.
6. `nao-motion calibrate` (section précédente), puis `nao-motion check-config`.
7. Ajuster vitesses, accélérations, raideurs (`robot.yaml` → `stiffness`) et contraintes manuelles entre membres (`guardian.yaml`).
8. `nao-motion ui --adapter bridge`. Faites un premier essai avec des gestes lents, une main près du bouton ARRÊT, et un robot dans une posture où un relâchement ne le ferait pas tomber.

Points à vérifier sur NAOqi 2.1.4, que ce dépôt ne présume pas : le comportement de `setAngles` / `killTasksUsingResources` en maintien, la raideur minimale qui permet un guidage confortable, et la latence réelle du streaming à 25 Hz.

---

## Émotions et intentions

`configs/emotions.yaml` propose une liste **courte et modifiable** : joie, tristesse, colère, peur, surprise, calme. Ce ne sont pas des catégories scientifiques figées, juste des étiquettes de travail. Les **intentions** (accueillir, refuser, saluer, montrer) sont séparées des émotions : on peut accueillir avec joie ou avec calme. Elles sont enregistrées comme annotation des prises mais ne conditionnent pas encore les modèles. Modifier la liste d'émotions rend les anciens checkpoints incompatibles, et ils sont refusés explicitement.

## Limites connues de ce socle

- Les données et les démonstrateurs sont synthétiques. Aucune mesure d'expressivité réelle.
- Le pont NAOqi et l'adaptateur `bridge` sont testés uniquement contre un backend factice.
- L'alignement temporel est linéaire (pas de DTW). Les variantes ne sont pas encore apprises.
- Les contraintes de collision sont linéaires et indicatives. Pas de modèle géométrique du robot.
