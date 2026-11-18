<h1>
  <img src="./assets/nao-emotion-motion-icon.png" alt="" width="48" height="48" align="middle">
  nao-emotion-motion
</h1>

![NAO en mouvement, avec des trajectoires lumineuses autour de la tête et des bras](./assets/nao-emotion-motion-social.jpg)

# nao-emotion-motion

Learning and generating expressive **arm**, **head** and **hand** movements for NAO from physical demonstrations, conditioned on an **emotion** and an **intensity** (0 to 1).

The motion component is not an LLM. It is a small supervised neural model (PyTorch) with lightweight inference that can also run on numpy alone. An external LLM can supply the emotion and intensity, but it only ever gets a movement **proposal**: execution remains an explicit operator action.

> **Status: first foundation, validated in simulation only.** The synthetic data is there to check the technical pipeline. It says nothing about expressiveness. Nothing has been tested on a real robot yet.

> **Note on language:** the web interface and the emotion/intention identifiers are in French. Labels from the interface are quoted as they appear, with an English gloss. Emotion IDs: `joie` (joy), `tristesse` (sadness), `colere` (anger), `peur` (fear), `surprise` (surprise), `calme` (calm).

---

## Quick start (no robot)

```bash
pip install -e ".[dev]"          # numpy, pyyaml, torch (+ pytest)
nao-motion check-config          # checks configs/ and lists what is missing for the real robot
nao-motion demo                  # full walkthrough in simulation (~15 s), in ./demo_sim
cp .env.example .env             # robot identity and address (read by the config and by the bridge)
pytest                           # tests
```

`nao-motion demo` runs through the steps of the first result:

0. calibration procedure exercised on the simulated robot (no real effect, not activated);
1. collection and storage of arm demonstrations (synthetic demonstrator, simulated adapter);
2. inspection of a take and virtual replay validated by the guardian;
3. training of the arm model;
4. generation, validation, then execution of the full cycle on the simulated robot;
5. head collection, **independent of the arms**: the head is guided (here by a simulated demonstrator) for each emotion, intensity and event, while the arms stay stiff in place;
6. head training on its own, then simulated execution of the combined arms + head gesture with an event;
7. hand collection (opening driven from the interface, simulated), hand training, combined arms + head + hands gesture.

## Local collection interface

```bash
nao-motion ui                    # http://127.0.0.1:8765 (simulated adapter by default)
nao-motion ui --adapter bridge   # to the robot through the NAOqi bridge (see below)
```

The interface listens on `127.0.0.1` only. It has nine tabs:

- **1 · Bras** (arms): choose the emotion, the intensity, optionally an intention, and the **event** in the same field of view as the head tab (azimuth, elevation, distance, or no event). *Tirage aléatoire* (random draw) draws all of them together, favouring what has been recorded least (see *Balanced random draw* below). The event is stored with the take and given to the arm model, so an oriented gesture (pointing, shielding on one side, reaching out) must be made towards the direction and distance shown. The instruction is shown in large type (“surprise — 0,8 · événement 30° à gauche, 15° vers le haut, à 0,50 m”). Then: *Préparer* (prepare: releases the arms and holds the head), *Démarrer* (start: countdown, then recording of the **measured** angles and their timestamps), *Arrêter* (stop). A preview of the curves is shown without any movement, then you choose *Accepter* (accept), *Garder en attente* (keep pending), *Rejeter* (reject) or *Recommencer* (retry), with an optional annotation.
- **2 · Tête** (head): same instruction as the arms tab (emotion, intensity, **event** in the field of view, *Tirage aléatoire*), and **no arm movement is needed**: this pass is independent. In the field of view, the dot is placed by azimuth and elevation, and its size shows proximity (bigger = closer). Place it with a click or by typing azimuth, elevation and distance; the distance may stay unknown. The random draw picks azimuth, elevation and distance (log-uniform, so there are as many near events as far ones) within the ranges in `robot.yaml` (`events`), sometimes with no event if you allow it. *Préparer* (prepare) releases the head and holds the arms stiff in place, with no motorised movement. *Démarrer* (start: countdown, then recording of the measured head angles), *Arrêter* (stop). What is recorded: emotion/intensity, event, head.
- **3 · Mains** (hands): opening of NAO's fingers, `LHand` and `RHand`. It is not on/off: every value from **0 (closed)** to **1 (open)** is possible. NAO's hands cannot be moved by hand, so they are driven from this tab: one slider per hand, *Fermer* / *½* / *Ouvrir* buttons per hand and for both, an option to link the sliders, and keys in this tab (**↑ / ↓** open / close both hands by 0.1, **O** opens both, **F** closes both). Outside a recording, each setting is a short motorised movement validated by the guardian (controller armed, *Mouvements autorisés*). The tab is also the **hand pass** of the collection, independent of the arms: same instruction as the other tabs (emotion, intensity, **event** in the field of view, random draw). The event position matters here too: an event on the robot's right can open the right hand and close the left one. Arms and head stay stiff in place, nothing is released, and you set the hand opening **live** until *Arrêter*. Your setting is followed at a velocity and acceleration limited by the guardian; that filtered command and the measured opening are both recorded. Steps *Préparer* / *Démarrer* / *Arrêter* / *Raidir tout sur place* (keys 1 to 4).
- **Prises** (takes): distribution by emotion × intensity (accepted / total), filterable list, curves, virtual replay (animated cursor, smoothed replay overlaid), status and annotations.
- **Réseau** (network): settings of the next trainings started from the interface (hidden layers, dropout, weight decay, lr, smoothing, left/right mirror of takes without an event, number of cross-validation parts, number of B-spline control points, *peu de données* preset; empty field = `configs/model.yaml`, which is never modified; the last values typed are remembered by the browser) and a **diagram of the MLP** updated live: inputs (emotion one-hot, intensity, the 6 event features) → hidden layers (tanh, dropout) → outputs (K B-spline control points per joint + log duration) → B-spline decoding. One button per group (arms, head, hands) shows that group's network and its `model.yaml` defaults. Below: the number of parameters against the number of values to learn (accepted takes × outputs), as a rough guide (≤ 1 parameter per value learnt is comfortable, > 3 risks overfitting). The **Combien de points ?** button measures, on the accepted takes of the group, how much detail curves of 8, 12, 16, 24 and 32 control points lose (same as `nao-motion fidelity arms`); a row is clicked to use that number for the next trainings.
- **Cross-validation** (`training.folds` in `configs/model.yaml`, default 5; *Validation croisée* field in the Réseau tab; `nao-motion train … --folds N`): the takes are dealt into N parts (a take never sits on both sides), N fresh MLPs are trained, each validated on the part it never saw, so every take is validated exactly once. The validation shown is pooled over the parts, with the mean ± spread per part (chart *Validation croisée : chaque part* in the Entraînement tab). The **kept model** is then trained from scratch on **all** the takes for the mean best epoch of the parts. Training takes about N + 1 times longer. `folds` 0 or 1 = the former single 75/25 split (`val_fraction`).
- **Entraînement** (training): start arm, head or hand training. Results are listed in **chronological order** (oldest first; a button flips the order; filter by group), with the charts of the selected run on the right: validation ÷ mean gesture of every run over time; training / validation / reference RMSE against the target zone (validation ≤ 0.9 × the mean-gesture baseline); learning curve (train and validation loss, kept epoch); validation error per joint with the training error and the per-joint baseline; and a checklist of what to expect (validation clearly under the mean gesture, validation / training ratio ≤ 1.5, best epoch neither at the very start nor at the very end, guardian acceptance of generated gestures and of your takes, for information only). *Copier* copies all the metrics as JSON.
- **Field of view: robot view / observer view**: the ⇄ button above each field of view (or the **V** key) flips left and right **on screen only**; the recorded event never changes (positive azimuth = robot's left). *Vue robot* (robot view, default): you look the same way as the robot and see the back of its head in the middle, so the robot's left is on the left. *Vue observateur* (observer view): you face the robot and see its face, so the robot's left is on the right, like a mirror. The choice applies to every field of view and is remembered by the browser. The head pictures are `nao_motion/ui/static/nao-head-back.png` (back) and `nao-head-front.png` (front), square PNGs with a transparent background; while they are missing, the drawings `nao-head-back.svg` / `nao-head-front.svg` are used. The spoken instruction now says “à gauche **du robot**” to remove any doubt.
- **Balanced random draw**: in the *Bras*, *Tête* and *Mains* tabs, the draw favours what is missing from the takes of that pass (pending and accepted takes; rejected ones are to be redone). The space is cut into cells: emotion × intensity band (low / medium / high) × event zone (azimuth right / front / left × elevation low / high × distance near / far, plus “no event” when allowed, aimed at 20 %): 216 cells with the default settings. Each cell's score is its surplus over its fair share, weighted ×4, plus the surplus of its emotion, of its intensity band and of its zone, all counted in takes; a cell is drawn with probability ∝ exp(−2 × (score − lowest score)). In practice every cell gets one take before any gets a second, and among equally filled cells those of the emotion, band or zone lagging behind win, so the totals per emotion, band and zone stay within 1–2 takes of each other at all times. The choice among the least filled cells is random, and the exact values inside the cell (intensity to 0.05, angles to the degree, log-uniform distance) are drawn uniformly. The cell aimed at and its count are shown under the button. Settings: `robot.yaml` → `events.balance` (`strength: 0` gives back the plain uniform draw). The *Génération* tab has no takes to balance: its draw is uniform over the cells.
- **Génération** (generation): choose an emotion, optionally a blend of two emotions, the intensity, a variation, the temperature of a CVAE model and the event (same field of view, same random draw). The *Espace latent (CVAE)* section shows the map of z and a sweep of one z dimension (see [CVAE mode](#cvae-mode-learned-variants)). You get the curves, the guardian report on the gesture and on the full cycle, and an *Exécuter* (execute) button (disabled if the cycle is rejected). Motor stiffness is handled in the tab: *Play on the robot* stiffens the joints of the cycle in place just before its first command (other motors are left as they are) and, with *relax the motors at the end* (ticked by default), relaxes them once the cycle has ended normally (not after **ARRÊT**, which holds the posture). *Repos puis relâcher* (key **3**, armed motors) goes back to rest from the measured posture, stiffening relaxed joints in place first, then relaxes every motor (arms, head, hands). *Raidir sur place* (key **4**) stiffens every motor on its measured posture with no movement. *Relâcher* (key **Space**) relaxes every motor at any time, **including mid-gesture**: the motion ends there, with no fault and no need to acknowledge (hold the arms).
- **Robot**: *Connecter* (connect, no movement), *Acquitter* (acknowledge), *Raidir sur place* (stiffen in place), *Retour au repos* (return to rest, explicit), *Relâcher* (release).
- **Calibration**: see “Guardian calibration” below.

The **ARRÊT** (stop) button interrupts recording, replay and execution. The posture is held and the controller is disarmed. There is **no** automatic return to rest.

**Keyboard shortcuts** (shown under the tab bar, ignored while typing in a text, number or list field, and with Ctrl, Alt or Cmd):

- **Space** releases the limbs, anywhere in the interface: the arms in the *Bras* tab, the head in the *Tête* tab, the selected group while a calibration is open, every motor (arms, head, hands) in the *Génération* tab, even during *Play* (the motion ends there), the arms and the head elsewhere. Like *Préparer*, it asks for no confirmation: a warning on the *Bras* and *Tête* tabs reminds you to hold the limbs before pressing. Releasing commands no movement.
- **1 2 3 4** run the take steps of the *Bras*, *Tête* and *Mains* tabs: *Préparer*, *Démarrer*, *Arrêter*, *Raidir sur place*. In the *Génération* tab they are *Générer et valider*, *Play on the robot*, *Repos puis relâcher*, *Raidir sur place*. On an AZERTY keyboard, the top-row keys work with or without Shift, as does the numeric keypad.
- **→** (right arrow) runs the next step, deduced from the recording state; that step is outlined. The keys press the same buttons, with the same server-side checks.
- **Enter** accepts the take shown in preview (*Accepter* button) in the *Bras*, *Tête* and *Mains* tabs, with no confirmation. It also works from the take's *Annotation* field: type the note, then Enter. In these tabs, Enter never presses the button that kept the focus.
- **0** runs the *Tirage aléatoire* (random draw) button of the *Bras*, *Tête*, *Mains* and *Génération* tabs: emotion, intensity, angle and distance are drawn at random, favouring what has been recorded least. Like the button, it only fills in the instruction and the event: no movement. The draw keeps the boxes as they are: if *événement présent* (event present) is unchecked in the tab (or *utiliser l'événement* in *Génération*), only the emotion and the intensity are drawn, still balanced, and the event stays absent. On an AZERTY keyboard, the top-row key works with or without Shift, as does the numeric keypad.
- **V** flips the field of view between the two viewpoints, in the same tabs (same as the ⇄ button).

In simulation, a synthetic demonstrator replaces your hand. The takes it produces are marked `synthetic: true`.

### External entry points (HTTP JSON)

| Route | Purpose |
|---|---|
| `POST /api/event` | Event from the camera/microphone component: `{"present": true, "azimuth": 30, "elevation": 5, "distance_m": 1.2, "unit": "deg", "frame": "torso" \| "head", "source": "camera"}` (`distance_m` in metres, optional). With `frame: "head"`, include `head_yaw_rad` and `head_pitch_rad` at measurement time; otherwise the angles read on reception are used. `{"present": false}` signals that there is no event. `ttl_s` is optional (past it, the event is treated as absent). |
| `POST /api/request` | For an LLM: `{"emotion": "joie", "intensity": 0.7}` or `{"mix": {"joie": 0.5, "surprise": 0.5}, ...}`. The response contains the trajectory, the guardian report and a `plan_id`. **Nothing is executed.** |
| `POST /api/execute` | `{"plan_id": ..., "confirm": true}`, with the controller armed: operator action. |

## Command line

```bash
nao-motion synth                        # fills data/ with SYNTHETIC takes (technical tests)
nao-motion takes --stats                # list and distribution of takes
nao-motion train arms                   # trains the arms ("accepted" takes)
nao-motion train head                   # trains the head on its own (no arm model needed)
nao-motion train hands                  # trains the hands on its own
nao-motion generate --emotion joie --intensity 0.7 --event-az 30 --event-dist 1.2 --out gesture.json
nao-motion generate --mix '{"joie":0.6,"calme":0.4}' --intensity 0.5 --variation 0.03 --seed 1
nao-motion generate --emotion peur --intensity 0.9 --simulate     # execution on the SIMULATED robot
nao-motion execute gesture.json --adapter bridge --confirm        # the only CLI path to a real robot
nao-motion robot-info                   # identity, NAOqi, factory limits, measured posture (read-only, via the bridge)
nao-motion calibrate                    # guardian calibration procedure (see "Calibration")
nao-motion calibrate --adapter sim      # the same procedure on the simulated robot
nao-motion calibration-activate configs/calibration/<file>.yaml
nao-motion capture-pose --group left_arm --name start --adapter bridge   # reads the sensors, sends nothing
nao-motion serve-inference --port 8766  # remote inference (generates, does not execute)
nao-motion export-guardian --out guardian_export   # standalone guardian for another project (see "Standalone guardian")
nao-motion export-generator --out gesture_export   # trained gesture generator for another project (see "Gesture generator library")
nao-motion export-dataset ../nao-emotion-motion-dataset --author "Name"   # dataset to publish (see "Sharing the dataset")
nao-motion import-dataset ../nao-emotion-motion-dataset   # takes, calibrations and models back into data/, configs/, runs/
```

---

## Architecture

```
configs/                versioned YAML: joints + limits, poses, emotions, guardian, robot, model
bridge/naoqi_bridge.py  NAOqi bridge (Python 2.7 or 3), the only file depending on NAOqi; runs on the robot by default
nao_motion/
  config/               loading and validation of configurations
  events.py             event context, frames, conversions
  trajectory.py         timestamped trajectory (s, rad)
  data/                 take storage, representation (B-spline), resampling
  models/               features, PyTorch MLP, checkpoints, numpy export, training sets
  training.py           independent arm / head / hand training, metrics, group-wise split
  engine.py             combined arms + head + hands generation (proposes, does not execute)
  guardian/             independent validation of the full path
  control/              controller (cycle, execution, stop), two-pass recording
  adapters/             common interface, simulation, NAOqi bridge client
  inference/            remote inference server and client
  ui/                   local web interface (standard library + HTML/JS)
  workbench.py          service layer shared by the UI, the CLI and the demo
  sim/synthetic.py      synthetic demonstrators (simulation only)
```

Data goes into `data/`, training runs into `runs/`. Both folders are ignored by git.

### Three independent models, one engine

| | Inputs | Output |
|---|---|---|
| **Arms** | emotion (one-hot, or weighted blend), intensity, event `[present, sin az, cos az, elevation, distance known, proximity 1/(1+d)]` | K control points × 10 joints + log(duration) |
| **Head** | same inputs | K control points × 2 joints + log(duration) |
| **Hands** (optional) | same inputs | K control points × 2 (opening 0..1) + log(duration) |

Each group is **collected and trained on its own**, in any order: no model reads another model's output, and no pass needs a recorded or generated arm gesture. The engine then gives the same instruction (emotion, intensity, event) to every available model and merges their trajectories into one gesture, which the guardian validates as a whole. A group without a model stays at its start pose (arms, head) or is not commanded (hands).

Timing (`model.yaml`, `combine.timing`): with `hold` (default), all groups start together at their own learned tempo, and a shorter group holds its final pose until the longest one ends; with `stretch`, every group is slowed down to the longest duration.

Trade-off: consistency between groups comes from the **shared instruction** (the arm points towards the event, the head looks at it, the hand on that side opens), not from a learned synchronisation. A fine timing link such as the head following the hand with a slight delay is no longer learned. A shared latent or seed across groups is a possible later improvement, and a unified whole-body network is planned (see [TODO](#todo)).

Head and hand takes recorded earlier on replayed arms (old format) remain valid: training reads only the head or hand joints. Head and hand **models** trained in the old format (which took the arms as input) are rejected on reload and must be retrained.

The target of the hand model is, by default, the command the operator gave (`hands_model.target: command`, the intention), not the hand sensor, which lags behind the motor (`measured` is available). Its control points are clipped to the 0..1 range, so the B-spline stays within it without a velocity break.

### Motion representation

- **Raw** takes (measured angles, real timestamps) are never modified.
- Each take has a **useful segment**. For an arm take, it is proposed automatically by trimming the stillness at the start and end, and it remains editable. For a head take, it is proposed the same way; for a hand take, from the opening command.
- The segment is mapped to a **phase** u ∈ [0, 1] by linear resampling over 100 points, then projected onto **K cubic B-spline control points** per joint (`representation.n_control_points`, default 12; was 8, which flattened beats, waving and quick peaks). K can be set per training (Network tab, `--control-points`) and is stored in each model, so models trained with another K keep working, and the exported `nao_gesture.json` carries each model's own K. `nao-motion fidelity <group>` shows what a given K loses on the takes: whatever the curve loses, no model can produce. With enough points, a generated gesture that is still too flat comes from the model, which predicts an average of the takes (few takes, several ways of showing the same emotion). The **duration** is learned separately, in log.
- **Motion energy** (`training.motion_energy_weight`, default 0.1; Network tab, `--motion-energy`): a loss term asking each joint to move as much as in its take (RMS speed along the gesture), whatever the timing. With the pointwise error alone, the best prediction is the MEAN of the takes, and waves done at different moments cancel out: a wave (« coucou ») becomes a still arm. The training report gives `val_motion_kept` (movement of the predictions / movement of the takes; 1 = as lively) next to the same figure for the mean gesture. With this term, the pointwise RMSE can exceed the mean gesture's: an oscillation shifted in time counts as an error, a still arm hardly does.

Why this choice: with little data, you need few output parameters. The resulting curves are smooth (continuous velocity and acceleration, which helps the guardian). The mapping between parameters and trajectory is linear, so interpolating between examples or between emotions stays consistent. This linear alignment does not register the sub-phases of a gesture (no DTW); that is a possible future improvement.

### Averaging of gestures and variants

The first model is a **deterministic regression**. If “joie” (joy) has been demonstrated in two different ways, it learns the **average** of the two, which may be a gesture never shown. The synthetic demonstrator deliberately contains such a case. For the first prototype, variants are obtained as follows:

- **interpolation**: continuous intensity and weighted blend of emotions (`--mix`);
- **variation**: smooth, reproducible noise (`--variation`, `--seed`) on the inner control points, not learned.

### CVAE mode (learned variants)

When the same instruction (emotion, intensity, event) was demonstrated in several **different** ways, the deterministic MLP returns their mean, which is flat. The **CVAE mode** (`configs/model.yaml`, section `cvae`; *Mode CVAE* box in the Réseau tab; `nao-motion train arms --cvae`) keeps the same MLP as a **decoder** and gives it one more input, a latent code **z** (`latent_dim`, 4 by default). A small **encoder** MLP reads each take during training and gives its z, so z ends up describing *which way of doing* a take shows, with no label to add. At generation, z is drawn from N(0, T²): **T** is the *Température* field of the Génération tab (`cvae.temperature`, 1 by default); **T = 0 gives the central gesture** (z = 0, always the same). The log-duration output is unchanged.

Off (the default) = the plain MLP, unchanged: same checkpoints, same numpy export, same standalone library. A CVAE model is exported the same way; the standalone library and any code that does not pass z play its central gesture.

Settings (Réseau tab or `model.yaml`): `latent_dim`, `beta` (weight of the KL term, 0.003; higher = a smoother z map but fewer z dimensions used), `kl_warmup` (epochs over which beta grows from 0, 200; the best epoch is only looked for after them, on the validation bound with the full beta), `encoder_hidden` ([64]). In CVAE mode `motion_energy_weight` is set to 0 unless given explicitly: the amplitude comes from z.

Training tab, for a CVAE run: **dimensions of z in use** (variance of the takes' codes > 0.01, chart per dimension), **reconstruction** (each take encoded then decoded; with cross-validation, by the fold model that never saw it), **random draws** (motion kept, and how far two draws are from each other compared with two takes). "Validation" and "training" stay those of the central gesture z = 0, comparable with a plain MLP.

Génération tab, *Espace latent (CVAE)*: the **map of z** (2D PCA of the takes' codes, one point per take coloured by emotion; dashed circles = 1σ and 2σ of the draws). Clicking a take plays it back through the model (its z, its instruction and event); clicking an empty spot generates a new gesture from that z with the instruction of the tab. The **sweep** draws, for each joint, the curves obtained when one z dimension goes from −r to +r, overlaid. Both are read only; playing still goes through *Générer et valider* and *Play*. Routes: `GET /api/latent?component=arms`, `POST /api/latent_sweep`, and `temperature` / `latent` / `event` in `POST /api/generate`.

Check on synthetic agitated takes (each emotion done in 3 different ways, 108 takes, 5-fold cross-validation): the plain MLP stays near the mean of the three ways (0.25 rad from the takes, like the mean gesture); the CVAE plays back takes it never saw at 0.07 rad with 87 % of their motion, uses 2 of its 4 z dimensions (3 ways need 2), and its random draws are about twice closer to one of the demonstrated ways than the MLP output (0.07 to 0.13 rad vs 0.16 to 0.18 rad). With 60 takes that are all different, the encoder cannot generalise (no structure to find): the CVAE needs takes that repeat a few ways of doing.

### Measure rather than impose a number of takes

Each training run writes to `runs/<run>/metrics.json`:

- the RMSE (rad) on dense trajectories in training and validation, overall and per joint;
- a comparison **baseline**: the average of takes with the same emotion and a nearby intensity. If the model does no better, it should be visible;
- the duration error;
- the rate of generated trajectories accepted by the guardian over an emotion × intensity × event grid;
- a warning when the data does not allow validation (fewer than two groups).

The training/validation split is done **by group**: each take is its own group, except old head or hand takes recorded on the same replayed arm gesture, which stay on the same side.

---

## Units, axes and frames

- **Angles**: radians. **Time**: seconds. Joint names: those of NAOqi.
- **Hands** (`LHand`, `RHand`): not an angle but an **opening**, 0 = closed, 1 = open (factory range of NAO V4, read by `ALMotion.getLimits`). They are declared in `joints.yaml` (`hands`), outside the calibrated groups: the calibration does not ask for them and they stay bounded by this range. Guardian margin, velocity and acceleration of the hands are set per joint in `guardian.yaml` (`per_joint_margin`, `per_joint_velocity_rad_s`, `per_joint_acceleration`), and their rest and start openings in `poses.yaml` (`hands`). The cycle is not refused because the hands are open at rest: they join the start pose with the rest of the body.
- **Common event frame**: NAOqi `FRAME_TORSO`. x points forward, y to the robot's left, z up.
  - azimuth = atan2(y, x), **positive to the left**;
  - elevation = atan2(z, √(x²+y²)), **positive upward**.
- **Distance** (optional): in metres, from the torso to the event. The model receives the proximity 1/(1+d), which is 0.5 at 1 m and does not depend on any range fixed in advance. An unknown distance is flagged explicitly, not replaced by an arbitrary value. The ranges in `robot.yaml` (`events`) are only used for the random draw and the field of view.
- **No event**: `present: false`, encoded `[0, 0, 0, 0, 0, 0]`. An azimuth of 0 means “straight ahead” and is encoded `[1, 0, 1, 0, 0, 0]` if the distance is unknown, or `[1, 0, 1, 0, 1, 0.5]` at 1 m. These cases cannot be confused.
- A head model trained before distance was added (four event inputs) is rejected on reload and must be retrained. An arm model trained before the event was added to its inputs is also rejected and must be retrained. Arm takes already recorded keep the event that was current when they were recorded (often “no event”).
- **Conversion from the camera (head frame)**: rotation `Rz(HeadYaw) · Ry(HeadPitch)` with the head angles at measurement time. The neck translation is neglected, which remains reasonable beyond about 50 cm.
- **NAOqi conventions**: HeadYaw > 0 turns the head left, HeadPitch > 0 lowers it. Looking at an elevation *e* therefore corresponds roughly to HeadPitch = −*e*.
- Events are always stored in the common frame, with their source and timestamp.

## Data format

Each take is stored in `data/takes/<take_id>/`:

- `samples.npz`: `t` (N,) in s since the start of recording, `q` (N, J) in **measured** rad, `joints`; for a hand take, also the opening command sent (`ref_t`, `ref_q`, `ref_joints`), at the same timestamps as the measurements (old head and hand takes also hold the command sent to the replayed arms);
- `meta.json`: `take_id`, `session_id`, `kind` (`arms` | `head` | `hands`), `emotion`, `intensity`, `intention` (optional), `joints`, `unit`, `time_unit`, `guided_joints`, `driven_joints`, `event` (torso frame), `event_frame`, `segment`, `head_initial`, `hands_initial` (hand takes), `arm_source`, `arm_take_id`, `arm_generation`, `arm_gesture_id` (old format only: arm gesture replayed during a head or hand take, `null` for independent takes), `calibration_version`, `config_fingerprint`, `adapter`, `sample_rate_hz`, `synthetic`, `status` (`pending` | `accepted` | `rejected`), `annotations` (`notes`, `tags`), `created_at` and `format_version`.

Data is rejected at save time if a value is not finite, if timestamps are not strictly increasing, if the emotion is unknown, if the intensity is outside [0, 1], if the units are not the expected ones, if a joint is both guided and driven, or if an old-format head or hand take refers to an arm take that does not exist.

Checkpoints (`runs/<run>/model.pt`) contain the architecture, weights, joints, emotion list, normalisation, representation, calibration version, configuration fingerprint, training and validation takes, hyperparameters and metrics. Head and hand checkpoints are marked `independent: true`. On reload, a model is **rejected** if its emotions or joints differ from the current configuration, and a warning is issued if the calibration has changed. `model_numpy.npz` allows inference without PyTorch.

Where inference runs is set in `configs/robot.yaml` → `inference`: `location: local | remote`, `backend: torch | numpy`, `remote_url`. A trajectory computed remotely is always revalidated locally by the guardian.

### Sharing the dataset, moving to another PC

Takes (`data/`), models (`runs/`), calibration drafts and `.env` are not versioned in this repository. Saved calibrations (`configs/calibration/*.yaml`, `ACTIVE`) are not ignored but are not committed either. To keep the code repository light, the dataset is published as a **separate repository** (GitHub, or Hugging Face, which is also a Git repository):

```bash
nao-motion export-dataset ../nao-emotion-motion-dataset --author "Name"
cd ../nao-emotion-motion-dataset && git init && git add . && git commit -m "Dataset v1"   # then push
```

The export keeps only **accepted** takes recorded on the **real robot** (no `synthetic`), without the `incomplete` tag (options `--status`, `--include-synthetic`, `--include-incomplete`), with byte-for-byte copies of the takes, the calibrations they cite, `configs/emotions.yaml` and `joints.yaml`, the latest arm, head and hand models (`--no-models` to leave them out), the **ready-to-use gesture generator** built from those same models (`generator/nao_gesture.py` + `generator/nao_gesture.json`, the files of `export-generator`, with the guardian of the active calibration; `--no-generator` to leave it out), an `index.csv`, a dataset card (`README.md`), a `LICENSE` for the data (`--license CC-BY-4.0` by default; the code stays MIT) and a `dataset.json` manifest with the sha256 of every file. `--strip-notes` empties the free-text notes. Exporting again into the same folder updates it (new takes added, changed metadata rewritten, takes no longer selected removed); its `.git` is never touched, and a `README.md` edited by hand since the last export is kept (the updated card goes to `README.generated.md`). The export folder must be outside `data/`, and is best kept outside this repository. The generator lets another project take the trained model straight from the dataset repository; the dataset card explains how to use it. It is skipped, with the reason printed, when it cannot be built (no active calibration, no `model_numpy.npz`): the takes are exported anyway. Its files are rewritten only when the models or the calibration change, so a re-export without a new training leaves no diff. Takes recorded before the first calibration cite the placeholder name of `joints.yaml` (`placeholder-0`): they are counted as recorded without calibration, not reported as a missing calibration.

**What a push to the dataset repository contains, and how to make it lighter.** Only the selection above goes into the export: rejected, pending, simulated and interrupted takes, and every training except the active model of each group (`runs/latest_<group>.txt`), stay on the PC. The *Takes* tab shows which takes would be exported (column *Export*). To clean up anyway, takes (*Takes* tab, quick selection: rejected, pending, simulated, incomplete, not exported) and trainings (*Training* tab, results list, *all but active*) can be ticked and **moved to a trash**: `data/trash/takes/` and `runs/trash/`. A trashed item disappears at once from training, progress, coverage and export, and can be restored as it was. Trashing the active model of a group makes the newest remaining training of that group the active one (the page says so). Emptying the trash, all or selected items, is the only irreversible step and asks for confirmation. Once a take has been pushed to the dataset repository, the next export removes it from the folder, but it stays in that repository's Git history.

`import-dataset` does the reverse (it ignores `generator/`, rebuilt from the models at each export), for instance on a new PC after cloning the dataset repository: it checks the sha256 of every data file first (`takes/`, `calibration/`, `models/`, `configs/`; nothing is imported if one is missing or modified; `README.md`, `LICENSE` and `index.csv` may be edited by hand), copies the missing takes into `data/takes/`, adds the missing calibrations to `configs/calibration/` **without activating any**, and copies the models into `runs/` (sets `latest_<group>.txt` only if none exists). Nothing is ever overwritten.

To move everything to another PC without publishing, copying `data/`, `runs/`, `configs/calibration/` and `.env` into the clone on the new PC is enough: all paths are relative.

---

## Guardian and execution

The model **proposes**, the controller **supervises**.

- **Guardian limits**: they come only from a **calibration** measured on the robot (next section). The values in `joints.yaml` and `poses.yaml` are only used for simulation.
- **Guardian** (`nao_motion/guardian`): it accepts or rejects, never corrects. It rejects invalid formats, non-finite values, unknown joints or leg joints, incompatible joint sets and out-of-range durations. It checks **min/max limits with margin** and linear **coupled constraints** (the envelopes measured by the calibration, plus any you add in `guardian.yaml`; none by default) along the **full path** sampled at 100 Hz, not just at the endpoints. It limits **velocity** and **acceleration** per joint and checks the gap between the first point and the measured state. When velocity is exceeded, it reports the slowdown that would suffice, without applying it.
- **The guardian protects playback, never training.** It never removes a take from the dataset, never enters the loss and never picks the kept epoch. The training report shows two numbers for information: the share of generated gestures the guardian accepts (as played and raw), and the share of **your recorded takes** it accepts, as recorded and once made playable, with how far they go beyond the zone (median and max, in rad). After extending the calibration with the takes, most takes still fail *as recorded* by design: the zone keeps the calibration margin plus the guardian margin inside the measured extremes (about 0.08 rad), so the takes that defined an extreme touch it; what matters is the playable rate and an excess of a few hundredths of a radian. If it refuses most of your own takes, the guardian is stricter than the gestures you actually do (calibrated zone too narrow, speed limits below your gestures): any model that imitates them will be refused too, so fix the calibration (see *Extend with the takes*), not the model.
- **Speed limits** (`guardian.yaml`). Each joint's velocity limit is, by order of precedence: an explicit value in `per_joint_velocity_rad_s`; else `velocity_fraction_of_factory` (1.0: factory limits) × its factory no-load speed (`factory_velocity_rad_s`, from the Aldebaran motor documentation: 8.29 rad/s for HeadYaw, ShoulderPitch, ElbowYaw; 7.19 for HeadPitch, ShoulderRoll, ElbowRoll; 24.6 for WristYaw); else `max_velocity_rad_s`. The speeds measured on the takes (calibration `kinematics`) then raise them where they are higher, never lower them. Ceiling in every case: `robot.yaml` `max_fraction_speed` × factory speed, the speed `setAngles` leaves to the motor while streaming (above it the robot could not follow a validated gesture). Default acceleration: 30 rad/s² (head: 8). The Génération tab's guardian diagnosis lists each limit with its source.
- **Gestures are made playable before the check** (`nao_motion/guardian/playable.py`, `robot.yaml` → `execution.adapt_generated`, on by default). A generated gesture is first slowed down just enough for the velocity and acceleration limits (same path, longer time, never faster): by default only the moments that exceed a limit are slowed down, with a smooth time warp, so the rest keeps the tempo of the model (`execution.slow_down: local`; `uniform` slows the whole gesture by one factor). A joint that goes past its bound (or a violated coupled constraint) is brought back with a smooth saturation (no kink); joints that stay inside are left exactly as the model made them. The guardian then validates the result as usual and stays the only judge: what still fails is refused and nothing is sent. The Génération tab shows what was changed and the guardian verdict on the raw gesture. The standalone exports (`nao_guardian.py`, `nao_gesture.py`) do not do this adaptation yet.
- **Margins** (`guardian.yaml`): the zone is what the calibration reached, minus `zone_margin_rad` (0.02 rad), and `limit_margin_rad` (0.03) only keeps away from the factory stops. The calibration's own margin (0.05) is given back for the guardian, so it no longer stacks with the guardian's (0.08 rad were taken off each side of what the robot reached). The pair envelopes are widened the same way. Without `zone_margin_rad`, the old rule applies.
- Independent min/max limits and a few linear constraints **do not guarantee the absence of collisions**. `guardian.yaml` no longer ships example constraints: they were never measured and alone refused about one gesture in ten (any arm raised high, for instance).
- **Nominal cycle** built by the controller, not learned: rest → start → gesture → start → rest. The rest ↔ start transitions are minimum-jerk. The rest pose may lie **outside the motion zone** (for example arms resting on the knees, a contact posture): it is bounded only by the factory limits, and the rest ↔ start segments are the only ones allowed to leave the zone. They are checked against the factory limits, a reduced velocity (`rest_velocity_rad_s`) and the acceleration limit, without margin or coupled constraints. Making sure the start pose is reachable from rest without contact is the responsibility of whoever calibrates. The start ↔ gesture transitions are quintic and matched in velocity and acceleration, capped by each joint's own limits; among the durations within the caps, the shortest one that stays inside the guardian zone is kept (a longer transition into a moving gesture overshoots). The **entire cycle** is validated before the first command is sent, then revalidated against the measured state just before execution. It is rejected if the robot is not measured at rest (unless `from_pose="current"` is explicitly chosen).
- **Streaming execution** (setAngles at 25 Hz). At each step, the controller checks for a stop request, the connection (heartbeat) and the tracking error between command and measurement.
- **Joint roles**: a joint is *guided* (zero stiffness, read only), *driven* (stiff, commanded) or left as is. A command to a guided joint is rejected before reaching the robot: a released axis is never assumed to follow a command.
- **Stop and errors**:
  - *ARRÊT* (stop): execution is interrupted, the **measured** posture is held and the controller is disarmed.
  - *Error during execution*: the controller tries to hold the posture, goes into `FAULT` and waits for the operator to acknowledge.
  - *Connection loss*: nothing more can be sent from the PC. The **bridge watchdog**, on the robot side, then stops tasks and holds the posture if no heartbeat arrives for 0.6 s or if the client disconnects.
  - There is **never** an automatic motorised return to rest nor an automatic release. *Retour au repos* (return to rest) and *Relâcher* (release) are explicit actions, because the right reaction depends on the actual state of the robot.
- **No automatic movement** at startup, during tests or at the end of training. The controller starts disarmed, and every movement requires arming followed by an explicit action. Several tests check this.
- **Real robot**: arming is refused without an active calibration, or if the connected robot does not carry the name from the calibration or from `.env`.
- **Reboots observed during some movements**: their cause is not established here. The guardian **does not fix them** and must not be presented as a protection against them. To study them, the movements involved should be logged (trajectory, velocities, temperatures and current if available) and compared.

---

## Guardian calibration

Calibration is a procedure in its own right, done robot by robot. Its result is a versioned file, `configs/calibration/<robot>-<date>.yaml`. The active file is designated by `configs/calibration/ACTIVE`, and a calibration is never overwritten.

It is run from the **Calibration tab** of the interface (`nao-motion ui`), or in the terminal with `nao-motion calibrate`. Both use the same procedure and the same computation.

```bash
nao-motion ui                   # Robot tab: connect the robot, then Calibration tab
nao-motion calibrate            # the same procedure in the terminal, via the bridge; nothing is commanded
```

In the Calibration tab, you choose a group and proceed step by step, in any order. The table shows, live, the measured angles and the min/max widening during exploration, as well as the captured poses. For each joint pair, a scatter plot with its convex hull shows the areas not yet explored. Buttons are enabled only when the step is possible: computation waits until all groups are explored and all poses captured, activation waits for saving. Opening a calibration disarms the controller, and collection, execution and arming are refused while it remains open. Closing does not return anything to rest. In simulation, a synthetic demonstrator does the exploration, and a simulated calibration cannot be activated.

1. **Identification**: robot name, NAOqi version, hardware configuration, and factory limits read by `ALMotion.getLimits`.
2. **Guided exploration**, group by group (left arm, right arm, head). The group being explored is released and only read; the others are stiffened in place. Slowly bring each joint to its *useful* end stops, and go through the combinations (head turned **and** lowered, shoulder raised **and** spread, etc.).
3. **Rest and start poses**, captured for each group by placing the robot by hand (average of 5 readings).
4. **Computation**: the limits are the measured range, intersected with the factory limits, minus a margin (`configs/calibration.yaml`). If the exploration goes beyond a factory limit (mechanical play, a forced stop), the calibration is not rejected: the factory limit is kept, and a warning is shown and stored in the file (`warnings`). The Calibration tab shows the factory range of each joint during exploration. The **coupled constraints** come from the convex hull of each explored pair (`pairs`), simplified conservatively. The procedure is rejected if the exploration is too short or too narrow, or if a pose falls outside the hull.
5. **Review** of the summary, then save and activate (tab button, or `nao-motion calibration-activate`). From the interface, activation reloads the configuration without restarting, and the controller stays disarmed.

**Saved calibrations.** The top of the Calibration tab lists every file of `configs/calibration/`, including those from earlier sessions, with an *Activer* button; no calibration needs to be open. A file that cannot be activated says why (simulated calibration on a real robot, missing joint, pose outside the limits with the current configuration, unreadable file). The file is checked against the current configuration *before* `ACTIVE` is written, and if reloading fails the previous activation is restored. Files written by earlier versions of the software still load (`schema_version: 1` is unchanged; the hands are outside the calibration).

**Progress is kept.** Explorations, captured poses and reused groups are saved after each step in `configs/calibration/drafts/<robot>.npz` (not versioned) and restored the next time the calibration is opened, even after a server restart. *Repartir de zéro* (start over) clears them. A group can also be **reused from a saved calibration** (*Reprendre ce groupe*): its limits, coupled constraints and poses are copied, so you can keep the arms and redo only the head. Exploring a reused group replaces the reuse. The resulting file says which groups were reused (`notes`, and `imported_from` in `measured`).

**Extend with the takes.** Your accepted takes were done by hand on the robot, like the exploration: they show where the joints actually go. The *Extend with the takes* panel (under the saved calibrations) computes a **new** calibration = the chosen calibration ∪ the accepted real, complete takes, with the same margin and the same rules as the computation above: per-joint ranges, and the convex hull of each pair (`pairs`) grown with the take points. **Factory limits stay the absolute ceiling**; a joint whose factory limits were not read is not extended. Optionally (checked by default), per-joint **velocity and acceleration limits** are taken from the takes: the peak of the fastest take on the joints moved by hand (every take must pass), never below the current limit. They are stored in the calibration file (`kinematics`) and, while that calibration is active, **raise** the `guardian.yaml` / factory limits where they are higher; they never lower them. *Preview* shows each pair with the current zone (dashed green), the extended zone (blue) and the take points outside the current zone (red), plus a table of ranges and speeds; *Save as new calibration* writes a new file (the base one is never overwritten) that you activate, or not, with *Activer*. Route: `POST /api/calibrations/extend {file, speeds, save}`.

An active calibration replaces limits, poses and calibration version throughout the pipeline (guardian, takes, checkpoints). The manual constraints in `guardian.yaml` are added to it, in particular between limbs (arms ↔ head, arm ↔ arm), which a joint-by-joint exploration does not measure. The convex hull assumes a convex reachable region for each pair; an L-shaped region must be split by hand.

## Standalone guardian (export for another project)

The guardian can be exported as two files, to act as a safety layer for another project (for example the NAOqi bridge of ia-agent, in Python 2.7) without depending on this repository:

```bash
nao-motion export-guardian --out guardian_export
#   guardian_export/guardian.json    limits, velocities, coupled constraints, poses (active calibration)
#   guardian_export/nao_guardian.py  Python 2.7 / 3 checker, standard library only
```

The export is **refused without an active calibration**: the fallback limits are not those of the robot. `--allow-uncalibrated` forces a test export, which the checker then refuses to load without `allow_uncalibrated=True`.

`guardian.json` (`kind: nao_motion.guardian_export`, `schema_version: 1`, angles in radians, time in seconds) contains: the source (calibration version, robot, configuration fingerprint), the groups, per-joint limits and the margin, maximum velocities (default and per joint), maximum acceleration, check frequency, duration range, start tolerance, coupled constraints `sum(coef × angle) ≤ bound` (origin `calibration` or `manual`), and the rest and start poses. Legs are rejected.

```python
from nao_guardian import Guardian
g = Guardian.load("guardian.json")
rep = g.check_pose({"HeadYaw": 0.3, "HeadPitch": -0.1})                 # a target posture
rep = g.check_move(measured_posture, target, 1.5)                       # linear move in 1.5 s
d = g.min_duration(measured_posture, target)                            # minimum allowed duration
rep = g.check_trajectory(t, joints, q, start_state=measured_posture)    # full trajectory
if not rep.ok:
    print(rep.summary())     # rep.violations: code, joint, time, value, limit
```

On the command line: `python nao_guardian.py guardian.json gesture.json` prints the verdict and returns 0 (accepted), 1 (rejected) or 2 (usage error). The gesture file uses the `nao-motion generate` format (`{"joints", "t", "q"}`, optionally inside `{"trajectory": ...}`).

The checker gives the same verdicts as the project's guardian (compared on random trajectories in the tests). It only judges: it neither modifies nor slows down any movement, and commands nothing. Python 2.7 compatibility is checked statically (syntax and imports); it has not yet been run on the robot's Python 2.7.3. After a new calibration, redo the export.

## Gesture generator library (export for another project)

Once training is done, the trained models are delivered to another project (for example ia-agent) as **two files**, with nothing to install:

```bash
nao-motion export-generator --out gesture_export
#   gesture_export/nao_gesture.py    generator + guardian, Python 2.7 / 3, standard library only (no PyTorch, no numpy)
#   gesture_export/nao_gesture.json  the arms / head / hands models, their settings and the guardian (active calibration)
```

**Integration guide** (for an LLM or coding agent wiring the generator into another project: stiffness at the start and end of a gesture, rest and start poses, `from_rest` / `return_to`, guardian refusals, failure handling): [docs/generator-integration.md](docs/generator-integration.md).

The weights come from the numpy export written by every training run (`model_numpy.npz` of the latest run of each group), so the export does not need PyTorch. A group without a model is exported as missing: arms or head then stay in the start position, hands are not commanded. Like the guardian export, it is **refused without an active calibration** (`--allow-uncalibrated` for tests). After a new training or a new calibration, redo the export.

```python
from nao_gesture import GestureGenerator
gen = GestureGenerator.load("nao_gesture.json")
print(gen.emotions)                       # e.g. ['joie', 'tristesse', 'colere', 'peur', 'surprise', 'calme']
g = gen.generate("joie", 0.7,                                   # emotion (or a mix {"joie": 0.7, "surprise": 0.3}), intensity 0..1
                 event={"azimuth_deg": 30, "elevation_deg": 0, "distance_m": 1.2},   # optional; None = no event
                 current_pose=measured_posture)                 # recommended: adds the transitions
if g.ok:                                                        # accepted by the guardian
    names, angles, times = g.angle_interpolation()
    motion.angleInterpolation(names, angles, times, True)       # ALMotion, absolute angles
else:
    print(g.report.summary())                                   # never send it
```

- **Event**: torso frame (FRAME_TORSO), azimuth positive to the robot's LEFT, elevation positive UP, 0 = straight ahead, in degrees (`azimuth_deg`, `elevation_deg`) or radians (`azimuth_rad`, `elevation_rad`); `distance_m` is optional. A direction seen by the camera can be given with `"frame": "head"` plus the HeadYaw / HeadPitch measured at detection time (`head_yaw_rad`, `head_pitch_rad`).
- **Output**: `g.joints`, `g.times` (s, from 0), `g.positions` (one row per time step, radians; LHand / RHand are openings 0..1), `g.report` (guardian verdict on everything returned), `g.segments`, `g.notes`, `g.to_dict()`. `angle_interpolation()` drops the t = 0 point (NAOqi starts from the current posture), can thin out the points (`rate_hz=10`), and refuses a rejected movement.
- **Transitions**: the learned gesture starts from the calibrated start position. With `current_pose` (measured `{joint: value}`), the generator builds and validates the whole movement like the project UI: current posture → start → gesture → start (`return_to="rest"` goes back to rest). `from_rest=True` when the robot is at rest, which may be outside the movement zone (contact): that segment then follows the rest rules (factory limits, reduced velocity). Without `current_pose`, only the gesture is returned and the caller must bring the robot to the start position first.
- `variation=0.05, seed=...` adds a small random variation (0 = the learned gesture). Its random generator differs from the project's numpy one: same seed, different variant.
- `nao_gesture.py` contains the same guardian as `nao_guardian.py` (`Guardian`, `Report`, `check_rest_transition`...), so a separate guardian export is not needed. Command line: `python nao_gesture.py nao_gesture.json joie 0.7 30 0 1.2` prints the verdict (code 0 accepted, 1 rejected, 2 usage).

The tests compare the exported library with the project: same gestures as `MotionEngine` (numpy backend), same transitions and verdicts as the controller's `plan_cycle`. Python 2.7 compatibility is checked statically (syntax and imports) only; it has not been run on a real Python 2.7 yet. Nothing is ever sent to the robot by the library: executing the gesture is the integrating project's job.

## Connecting the real robot

Identity recorded on Monolithoid's NAO (in `.env.example`): **naoH25, head/body/arms VERSION_40 (NAO V4), NAOqi 2.1.4.13, Python 2.7.3, name “nao”**. To record or check it on another robot, over SSH (`ssh nao@<IP>`):

```bash
qicli call ALSystem.systemVersion        # NAOqi version
qicli call ALSystem.robotName            # name (tied to the calibration)
qicli call ALMotion.getRobotConfig       # model and head/body/arm versions
qicli call ALMotion.getLimits HeadPitch  # [min, max, max velocity, max torque], per joint
python --version                         # embedded Python
```

Steps:

1. `cp .env.example .env` (not versioned): expected robot identity, bridge port. These values override those in `configs/robot.yaml`.
2. **Choose where the bridge runs.** By default (`NAO_BRIDGE_LOCATION=robot`), it runs **on the NAO**, and there is **nothing to install**: the `naoqi` module is already present in the robot's Python 2.7.3. This is also what ia-agent does (embedded `nao_ws_bridge.py`). The `pc` mode requires the NAOqi Python SDK (pynaoqi, Python 2.7), which has become hard to find and install. The bridge tries `from naoqi import ALProxy`, then `import qi` (`--api` to force one).
3. **Prepare the robot's position.** The legs stay out of scope and are never commanded. Put the robot in a stable posture, for example sitting or crouching. Handle autonomous life, breathing and idle movements from your usual tools: the bridge does not touch them, because disabling them can make the robot move depending on the version.
4. **Start the bridge**, which sends no movement at startup:
   ```bash
   # on the robot (recommended)
   scp bridge/naoqi_bridge.py nao@<IP>:/home/nao/
   ssh nao@<IP> python /home/nao/naoqi_bridge.py --on-robot   # local NAOqi, listens on network port 9600
   # or on the PC, with the pynaoqi SDK (NAO_BRIDGE_LOCATION=pc)
   python2 bridge/naoqi_bridge.py            # address supplied later by the interface
   ```
   The bridge has no authentication: on the robot, use it only on a trusted network, and stop it (Ctrl+C) after the session. To test the protocol without a robot: `python bridge/naoqi_bridge.py --fake`. The bridge accepts only one client at a time (interface **or** command).
   Then, in `nao-motion ui --adapter bridge` → **Robot** tab, choose where the bridge runs, enter the **robot address (IP or DNS name)**, for example `192.168.1.10` or `nao.local`, and click *Connecter au robot* (connect to robot). This action sends no movement and changes no stiffness. The address is remembered in the browser. On the command line, `--robot <IP or DNS>` plays the same role, with `NAO_IP` as the default.
5. `nao-motion robot-info --robot <IP or DNS>`: checks the connection and shows identity, factory limits and measured posture, without any movement.
6. `nao-motion calibrate` (previous section), then `nao-motion check-config`.
7. Adjust velocities, accelerations, stiffness (`robot.yaml` → `stiffness`) and manual constraints between limbs (`guardian.yaml`).
8. `nao-motion ui --adapter bridge`. Make a first attempt with slow gestures, a hand near the ARRÊT (stop) button, and the robot in a posture where a release would not make it fall.

Things to check on NAOqi 2.1.4 that this repository does not assume: the behaviour of `setAngles` / `killTasksUsingResources` when holding, the minimum stiffness that allows comfortable guiding, and the actual latency of 25 Hz streaming.

---

## Emotions and intentions

`configs/emotions.yaml` provides a **short, editable** list: `joie` (joy), `tristesse` (sadness), `colere` (anger), `peur` (fear), `surprise`, `calme` (calm). These are not fixed scientific categories, just working labels. **Intentions** (`accueillir` welcome, `refuser` refuse, `saluer` greet, `montrer` show) are separate from emotions: you can welcome with joy or with calm. They are recorded as take annotations but do not condition the models yet. Changing the emotion list makes old checkpoints incompatible, and they are explicitly rejected.

## Known limitations of this foundation

- The data and demonstrators are synthetic. No real expressiveness measurement.
- The NAOqi bridge and the `bridge` adapter are tested only against a fake backend. The bridge needed no change for the hands (`LHand` / `RHand` go through `getAngles`, `setAngles` and `setStiffnesses` like any joint), but hand control has not been tried on the robot yet: hand speed under `max_fraction_speed`, and the hand tracking tolerance (`execution.hand_tracking_error`), are to be checked there. A hand kept stiff and closed for a long time heats up.
- Temporal alignment is linear (no DTW). Learned variants need the CVAE mode, still to be tried on real takes.
- Collision constraints are linear and indicative. No geometric model of the robot.

## TODO

### One unified network for the whole body

Today arms, head and hands are **three independent models** (see [Three independent models, one engine](#three-independent-models-one-engine)). This is a practical choice, not a design goal: guiding one body part at a time is much easier for the operator during collection. The longer-term target is a single network that generates arms, head and hands together, so that the coordination between parts (timing, the head following the hand, and so on) is learned rather than only implied by the shared instruction.

Planned path:

1. **Distillation.** Have the three current models generate thousands of complete gestures in simulation (emotion, intensity and event drawn with the balanced draw), then train a fourth, unified network to reproduce them. On its own this only copies the current combination: same quality, one file instead of three.
2. **Fine-tuning on whole-body takes.** Record real takes where the whole body moves at once, and fine-tune the distilled network on them. This is where the unified network can become better than the three separate ones.

Not to do: train a single network directly on the current takes. In each of them only one part moves while the others stay fixed, so the network would learn averaged, limp gestures.
