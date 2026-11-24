# Integrating the gesture generator into another project

This guide is written for an LLM or coding agent asked to make a NAO play the gestures of
nao-emotion-motion from another code base (for example a dialogue agent that decides
"joy, 0.7, a person on the left at 1.2 m"). It describes what the exported files do, what
they deliberately do **not** do, and the choices the integrating project has to make:
motor stiffness at the start and end of a gesture, rest and start postures, transitions,
guardian refusals.

Everything below is taken from the code of this repository:
`nao_motion/standalone/nao_gesture_template.py` (the generator), `nao_motion/standalone/nao_guardian.py`
(the guardian bundled into it), `nao_motion/gesture_export.py` (the export) and, for the
reference behaviour of the project's own UI, `nao_motion/workbench.py` and
`nao_motion/control/controller.py`. When this guide and the code disagree, the code wins.

Target robot so far: NAO V4 (naoH25, VERSION_40), NAOqi 2.1.4, Python 2.7.3 on the robot.
The library has been tested against the project in Python 3; Python 2.7 compatibility is
checked statically only (syntax and imports). It has **not** been run on a real robot yet.

## 1. What you get

```bash
nao-motion export-generator --out gesture_export
#   gesture_export/nao_gesture.py    generator + guardian, Python 2.7 / 3, standard library only
#   gesture_export/nao_gesture.json  arms / head / hands models, their settings, guardian limits and poses
```

- Copy **both** files into the other project, side by side. Nothing to install: no PyTorch,
  no numpy, no NAOqi import. The library can run on the PC or on the robot itself.
- The export is refused without an active calibration (`--allow-uncalibrated` exists for tests;
  such a file is then refused at load time unless `allow_uncalibrated=True`). Never use an
  uncalibrated file on the real robot: its limits are not those of the robot.
- The JSON is tied to one robot and one calibration (`gen.info["calibration"]`). After a new
  calibration or a new training, export again and replace both files together.
- The library **commands nothing**. It returns a validated trajectory; sending it, stiffening
  and relaxing motors, and stopping are the integrating project's job.

## 2. Minimal integration (NAOqi 2.1, Python 2.7)

```python
from naoqi import ALProxy
from nao_gesture import GestureGenerator, GestureError, GuardianError

gen = GestureGenerator.load("nao_gesture.json")       # load once, at startup
motion = ALProxy("ALMotion", robot_ip, 9559)

def measured_pose(names):
    # True = sensor values (not the last command)
    return dict(zip(names, motion.getAngles([str(n) for n in names], True)))

def play(emotion, intensity, event=None):
    names = gen.groups["arms"] + gen.groups["head"] + [j for j in gen.groups.get("hands", [])
                                                       if "hands" in gen.models]
    stiffen_in_place(names)                            # section 4
    pose = measured_pose(names)
    g = gen.generate(emotion, intensity, event=event, current_pose=pose,
                     from_rest=is_at_rest(pose), return_to="rest")
    if not g.ok:
        log(g.report.summary())                        # never send a refused movement
        return False
    names, angles, times = g.angle_interpolation()
    motion.angleInterpolation([str(n) for n in names], angles, times, True)   # blocking, absolute
    motion.setStiffnesses([str(n) for n in names], 0.0)                      # relax at rest (section 4)
    return True
```

`str(...)` around joint names: the library uses `unicode_literals`, so under Python 2.7 the
names are `unicode`. ia-agent's own bridge passes `str(joint)` to NAOqi; doing the same is the
safe choice (not verified on the robot).

## 3. API reference

### `GestureGenerator.load(path, allow_uncalibrated=False)`

Raises `GestureError` (a `ValueError`) for an invalid file, `GuardianError` for an
uncalibrated or invalid guardian section. Useful attributes:

| Attribute | Content |
|---|---|
| `gen.emotions` | emotion ids accepted by the models, in input order |
| `gen.groups` | `{"arms": [...], "head": [...], "hands": [...]}` joint names |
| `gen.models` | groups that have a trained model (`arms`, `head`, `hands`) |
| `gen.guardian` | the bundled `Guardian` (limits, poses, velocities) |
| `gen.info` | export date, calibration, source training runs; `capabilities` lists the optional `generate()` features of this library (`"duration_s"`, `"count"`) and `sequence` its margin settings. A caller that may load an older file pair checks `"duration_s" in gen.info.get("capabilities", [])` |
| `gen.timing` | `hold` or `stretch` (how groups of different durations are combined) |

### Emotions

The ids are English (`joy`, `sadness`, `anger`, `fear`, `surprise`, `calm`); the
interface shows French labels for them. Generators exported before the rename use the
former French ids (`joie`, `tristesse`, `colere`, `peur`, `surprise`, `calme`).

Always read the list from `gen.emotions` rather than hard-coding it: the list comes from
`configs/emotions.yaml` at training time, and a model refuses a different list.
`emotion` can also be a mix `{"joy": 0.7, "surprise": 0.3}` (non-negative weights, normalised).
An LLM that produces English labels needs its own mapping table to these ids.

`intensity` is a float in `[0, 1]`; anything else raises `GestureError`. The takes were
collected mostly between 0.1 and 1.0.

### Event (optional): where something is

`event=None` means "no event". Otherwise a dict:

| key | meaning |
|---|---|
| `azimuth_deg` or `azimuth_rad` | horizontal direction, **positive to the robot's LEFT**, 0 = straight ahead |
| `elevation_deg` or `elevation_rad` | vertical direction, **positive UP**, within [-90°, 90°] |
| `distance_m` | optional, strictly positive, metres |
| `frame` | `"torso"` (default, FRAME_TORSO) or `"head"` |
| `head_yaw_rad`, `head_pitch_rad` | required with `frame="head"`: head angles **measured at detection time** |
| `present` | `False` is the same as `event=None` |

Use `frame="head"` for a direction coming from a camera on the head (face or sound detection):
the library converts it to the torso frame with the head angles you measured at that moment
(neck translation neglected). The data were collected with azimuth in [-90°, 90°], elevation
in [-30°, 45°] and distance in [0.3, 3] m (`configs/robot.yaml` → `events`): values outside these
ranges are accepted but extrapolated.

What the models actually receive (for reference, `event_features()`):
`[present, sin(az), cos(az), elevation_rad, distance_known, 1/(1+distance_m)]`, appended to
`[emotion one-hot or mix..., intensity]`.

### `gen.generate(emotion, intensity, event=None, current_pose=None, from_rest=False, return_to="start", variation=0.0, seed=None, adapt=None, temperature=None, latent=None, duration_s=None, count=None, duration_fit=None, margin_k=None)`

Returns a `Gesture`. Raises `GestureError` for invalid inputs (unknown emotion, intensity out of
range, bad event, incomplete `current_pose`, bad `return_to`).

| Parameter | Effect |
|---|---|
| `current_pose` | measured posture `{joint: value}` for **every** joint of the gesture (arms, head, and hands if a hands model exists). Strongly recommended: with it, the move to the start pose is added when needed and the **whole** movement is validated. |
| `from_rest` | `True` when the robot is at its calibrated rest pose. The first segment is then checked with the rest rules (section 5). If the measured posture is farther than `2 × start_tolerance` (0.1 rad by default) from rest on any arm or head joint, the gesture is refused with a `not_at_rest` violation. |
| `return_to` | `"start"`: the movement ends in the calibrated start pose. `"rest"`: it continues from start to rest. Ignored without `current_pose`. |
| `variation`, `seed` | small random variation of the control points (0 = the learned gesture). Same seed = same variant (but not the same variant as the project's numpy generator). |
| `temperature` | CVAE models only (`gen.latent_dims` not empty): each call draws a latent code z ~ N(0, T²) per group, so the same prompt gives a different learned gesture (a "way of doing" seen in the takes). `None` (default) = the file's `latent_temperature` (model.yaml `cvae.temperature` at export; 0 for files exported before it), `0` = always the central gesture (z = 0). With `seed`, the draw is reproducible (not the same z as the project engine for the same seed). Recorded in `g.inputs["temperature"]` and `g.inputs["latent"]`. |
| `latent` | CVAE models only: a given z per group, e.g. `{"arms": [0.5, -1.0, 0.0, 0.2]}` (same z = same gesture as the project engine). |
| `adapt` | `None` (default): follow the `playback` settings exported with the JSON (the project's `execution` section), i.e. make the gesture playable as the project UI does (section 6). `False`: check the raw learned gesture only. |
| `duration_s` | fill this many seconds (typically the speech the gesture goes with) with **one** movement; see "Duration-targeted sequence" below. `None` (default): a single learned gesture, exactly as before. |
| `count` | force this many blended draws (with or without `duration_s`; without it, the whole material is played at the learned tempo). |
| `duration_fit` | `"intensity"` (tempo margin below) or `"off"` (learned tempo, the cut alone fits the duration). `None` = the file's setting. |
| `margin_k` | `k` of the tempo margin. `None` = the file's setting (1.0 by default). |

#### Duration-targeted sequence (`duration_s`)

With `duration_s=D`, the `gesture` segment lasts about `D` seconds, entry from and return to the
start pose included, and stays one continuous, validated movement:

1. **Draws**: learned gestures are drawn for the same emotion, intensity and event (CVAE models:
   a new latent z each; plain models: the same gesture repeated, unless `variation` > 0) until
   the entry time plus their learned durations add up to more than `D`. Each draw has its own seed derived from
   `seed` (the first draw uses `seed` itself, so it is the gesture `generate()` gives alone).
   Without `seed`, one is drawn and reported in `g.sequence["seed"]`: passing it back gives the
   same movement.
2. **Blend**: per group, the control points of the draws are concatenated into one cubic
   B-spline; the clamped end points at each junction are dropped, so the robot does **not** go
   back to the start pose between draws (the end of one draw and the start of the next are
   slightly smoothed). Within a draw, arms, head and hands share the draw's duration.
3. **Cut**: the curve starts on the start pose repeated, is cut on a control point and ends on
   the start pose repeated (departure and arrival at zero velocity and acceleration, see
   [gesture-curve.md](gesture-curve.md)). The cut is chosen on the duration **after** the usual
   playable adaptation (section 6), which may slow fast moments down.
4. **Tempo margin**: the result is played `s` times slower (`s` > 1) or faster (`s` < 1) to
   land on `D`, with `s` in `[max(scale_min, 1 - k (1 - i)), min(scale_max, 1 + k (1 - i))]`
   (`i` = intensity; file defaults `k = 1`, `scale_min = 0.7`, `scale_max = 3`, from the
   project's `model.yaml` → `sequence`). A faster tempo is kept only if the guardian accepts it,
   so velocities and accelerations never go past the limits. Intensity 1 = learned tempo: only
   the cut fits the duration (within about one knot span).
5. **Maximum**: `D` above the guardian maximum duration (`max_duration_s`, 15 s by default) is
   cut at that maximum (a note says so).

No transition is added around it: when the robot is already at the start pose, the movement
is the `gesture` segment alone and lasts about `D`, like the speech. `g.sequence` reports `draws`,
`seed`, `draw_seeds`, `draw_durations_s`, `target_s`, `cut_s` (whole movement before adaptation:
entry, learned time kept, return), `entry_s`, `return_s`, `playable_s` (after adaptation),
`time_scale` (`s`), `scale_range`, `duration_s` (of the `gesture` segment) and
`kept_control_points`. With CVAE models, `g.inputs["latent"]` then holds one z per draw for each
group. Files exported before this feature have no `sequence` settings: the defaults apply.

The movement returned with `current_pose` is made of named segments (`g.segments`, list of
`(name, start_s, end_s)`):

```
from_rest=False:                     gesture [-> start_to_rest]
from_rest=True:   rest_to_start   -> gesture [-> start_to_rest]
```

The `gesture` segment starts and ends on the start pose, at rest, inside its own B-spline
(with or without `duration_s`; without it, one learned gesture, uncut): there is no
`start_to_gesture` / `gesture_to_start` segment any more, so a sentence started with the
`gesture` segment starts with the movement. There is no `to_start` either: with
`from_rest=False`, the measured posture must already be within `start_tolerance` of the start
pose, otherwise the gesture is refused (`start_mismatch`, and a note in `g.notes`); bring the
robot back to the start pose first (e.g. with `rest_segment()`, see section 5). Explained with
diagrams in [gesture-curve.md](gesture-curve.md).

Without `current_pose`, only the `gesture` segment is returned. It starts on the start pose,
**not** from where the robot is: the caller must first bring the robot to the start pose (and
validate that move itself). A note in `g.notes` says so.

### `Gesture`

| Member | Content |
|---|---|
| `g.ok` / `bool(g)` | `True` only if the guardian accepted **everything** returned |
| `g.report` | guardian report on the whole movement: `.ok`, `.violations`, `.summary()`, `.to_dict()` |
| `g.gesture_report` | report on the learned gesture alone (diagnostic) |
| `g.joints`, `g.times`, `g.positions` | joint names; times in s from 0; one row per time step (rad; `LHand`/`RHand` are openings 0..1) |
| `g.segments` | see above |
| `g.duration` | total duration in s |
| `g.notes` | human-readable notes (missing model, robot not at the start pose...) |
| `g.group_durations_s` | learned duration of each group's gesture |
| `g.adaptation` | what was done to make the gesture playable: `changed`, `time_scale`, `max_shape_change_rad`, `joints_changed`, `notes` (`None` when `adapt=False`) |
| `g.raw_report` | guardian report on the raw learned gesture, before adaptation (diagnostic) |
| `g.sequence` | with `duration_s` or `count`: draws, cut and time factor (see above); `None` otherwise |
| `g.to_dict()` | everything, JSON-serialisable (good for logs) |
| `g.angle_interpolation(rate_hz=None)` | `(names, angles, times)` for `ALMotion.angleInterpolation(names, angles, times, True)`. Drops the t = 0 point (NAOqi starts from the current posture). `rate_hz=10` thins the ~50 Hz samples. **Raises `GuardianError` if the movement was refused.** |

A group without a trained model: arms or head are held at the start pose, hands are simply not
in `g.joints` (not commanded). Check `g.notes`.

### Command line (quick check)

```bash
python nao_gesture.py nao_gesture.json joy 0.7 30 0 1.2   # emotion, intensity, azimuth°, elevation°, distance m
python nao_gesture.py nao_gesture.json joy 0.7 --duration=8 --seed=1   # sequence filling 8 s (also --count=N)
# prints the verdict as JSON; exit code 0 = accepted, 1 = refused, 2 = usage
```

## 4. Motor stiffness: choices at the start and end of a gesture

The library never touches stiffness. On NAO, a joint with stiffness 0 does not follow
commands, and a joint relaxed while raised **falls under gravity** (arms drop, the head drops
forward). These are the options, and what the project's own UI does.

### Before the movement

The joints must be stiff before `angleInterpolation`, otherwise nothing moves (or moves weakly).

- **Stiffen in place (recommended, what the project does).** Read the measured angles, set the
  stiffness, and immediately command the measured angles so that there is no jump towards an
  old command (`nao_motion/adapters/base.py` → `set_roles`):

  ```python
  def stiffen_in_place(names, value=0.8):          # 0.8 = robot.yaml stiffness.driven
      names = [str(n) for n in names]
      q = motion.getAngles(names, True)
      motion.setStiffnesses(names, value)
      motion.setAngles(names, q, 0.05)              # hold where it is, slow
  ```

  Only the joints of the gesture are stiffened; other motors are left as they are. Measure
  `current_pose` **after** stiffening, just before `generate()`.
- **Already stiff** (the robot just finished a gesture with `return_to="start"` and stayed
  stiff): nothing to do, measure and generate.
- **Avoid `ALMotion.wakeUp()` and `ALRobotPosture`** for this: they also drive the legs (out of
  scope here, never commanded by this project) and go to NAOqi's own postures, not the calibrated
  rest and start poses. On a seated robot, `wakeUp()` can make it stand up.

### After the movement

| End | Stiffness after | When to use it | Caveats |
|---|---|---|---|
| `return_to="rest"` | **relax** (stiffness 0) | default of the project UI (Play, End "back to rest", "relax the motors at the end" checked; also key 3 "back to rest then relax") | rest is the calibrated rest, possibly in contact (arms on the knees): relaxing there is meant to be safe. The rest ↔ start segment is slow (see section 5). |
| `return_to="rest"` | stay stiff | several gestures in a row while keeping a calm resting look | motors heat when kept stiff for a long time; NAO may reduce stiffness on its own when hot |
| `return_to="start"` | stay stiff | gestures chained quickly (dialogue): next `generate()` with `from_rest=False` starts from the start pose | same heating; the robot holds its arms in the air between gestures |
| `return_to="start"` | relax | **not recommended**: the arms fall from the start pose | the project UI warns about this combination ("the arms drop: hold them") |

Relax with `motion.setStiffnesses(names, 0.0)` on the joints of the gesture only. Hands at
stiffness 0 stay where they are.

The project relaxes at the end **only when the movement ended normally**. After a stop (see
section 7) it **holds the posture** (no automatic motorised return to rest after an error),
unless the tracking stop action is `relax` (`robot.yaml` → `execution.tracking_stop_action`).
Doing the same is recommended.

### Autonomous behaviours

Autonomous life, breathing and idle movements can move the same joints during a gesture. This
project does not touch them (disabling them can itself move the robot depending on the version):
handle them in the integrating project, consistently with how it already uses the robot.

## 5. Rest pose, start pose and transitions

Both poses come from the calibration and are in the JSON: `gen.guardian.pose("rest")`,
`gen.guardian.pose("start")` (`{joint: rad}`).

- **Rest**: the robot's idle posture, chosen at calibration. It **may lie outside the movement
  zone** (arms in contact with the knees...). It is bounded only by the factory limits.
- **Start**: the posture every learned gesture starts from and ends in, inside the movement
  zone. How to reach start from rest without collision is the responsibility of whoever
  calibrated (there is no intermediate "exit" pose).

How each segment is built and checked (`generate()`):

| Segment | Built as | Checked against |
|---|---|---|
| `rest_to_start` (`from_rest=True`) | minimum-jerk (`rest_segment()`), ≤ 0.8 × per-joint velocity and acceleration limits when `rest_velocity_rad_s` is `factory` (default); with a number, ≤ 0.8 × that velocity and 0.25 × max acceleration | factory limits, rest velocity (rest rules): rest may be outside the zone; plus `not_at_rest` if not measured at rest |
| `gesture` | B-spline decoded from the models, the start pose repeated at both ends ([gesture-curve.md](gesture-curve.md)) | calibrated zone |
| `start_to_rest` (`return_to="rest"`) | minimum-jerk, same profile as `rest_to_start` | rest rules |

Practical consequences:

- Transitions to and from rest are **slow on purpose**: a 1 rad distance takes about 8 s at
  0.24 rad/s. If the robot must react fast, keep it stiff at the start pose between gestures
  (`return_to="start"`), or accept the delay.
- The first point of the movement must match the measured posture within `start_tolerance`
  (0.05 rad by default): measure `current_pose` immediately before sending, and do not let
  anything else move the robot between `generate()` and `angleInterpolation()`.
- To decide `from_rest`, use the same rule as the project UI ("auto" start: rest if measured
  there, else current posture):

  ```python
  def is_at_rest(pose):
      rest, tol = gen.guardian.pose("rest"), 2 * gen.guardian.start_tolerance
      hands = set(gen.groups.get("hands") or [])
      return all(abs(pose[j] - rest[j]) <= tol for j in pose if j in rest and j not in hands)
  ```

## 6. Guardian behaviour

The guardian bundled in `nao_gesture.py` is the `<guardian-core>` block of `nao_guardian.py`,
inserted at export time: same rules as the project guardian. It **only judges**: it never
modifies a movement, it accepts it or refuses it with a list of violations.

What it checks on the gesture and the transitions around it: known joints only, legs always
refused; per-joint limits measured at calibration minus a margin; coupled linear constraints
`sum(coef × angle) ≤ bound`; maximum velocity and acceleration (per joint); total gesture
duration within `[min, max]` (0.3 – 15 s by default); first point close to the measured state.
The path is checked on a fine grid (100 Hz by default), not only at its samples.

Violation codes you may see in `g.report.violations` (each has `code`, `message`, `joint`,
`time_s`, `value`, `limit`):

| Code | Meaning |
|---|---|
| `limit` | outside the calibrated limits (margin included) |
| `coupled` | a coupled constraint is violated |
| `velocity`, `acceleration` | too fast for that joint |
| `duration` | gesture shorter or longer than allowed |
| `start_mismatch` | first point too far from the measured posture |
| `rest_limit`, `rest_velocity` | rest / measured-posture transition outside factory limits or too fast |
| `not_at_rest` | `from_rest=True` but the robot is not measured at rest |
| `legs`, `unknown_joint`, `incompatible`, `format` | wrong input |

**Making a gesture playable (same as the project UI).** Before checking, the library does what
the project UI does (`nao_motion/guardian/playable.py`), following the `playback` section of the
JSON (exported from `configs/robot.yaml` → `execution`):

- control points projected into the zone (`project_control_points`), with a soft saturation
  near the limits;
- a local slow-down where a velocity or acceleration limit is exceeded (`adapt_generated`,
  `slow_down: local`), or a uniform one (`uniform`).

The guardian then checks the **adapted** movement; `g.raw_report` keeps the verdict on the raw
gesture and `g.adaptation` says what changed. A gesture can therefore last longer than learned.
`generate(..., adapt=False)` checks the raw gesture only. The UI's "Guardian off" test mode does
not exist in the library, and must not be reimplemented by bypassing `g.ok`.

What the guardian does not guarantee: absence of collision beyond the configured constraints,
electrical or software stability of the robot.

## 7. Failure handling

| Situation | What happens | Recommended reaction |
|---|---|---|
| Unknown emotion, intensity outside [0, 1], invalid event, incomplete `current_pose` | `GestureError` raised by `generate()` | fix the input (map LLM labels to `gen.emotions`, clamp intensity); log |
| Guardian refusal | `g.ok` is `False`, nothing to send; `angle_interpolation()` raises `GuardianError` | send **nothing**; log `g.report.summary()`. Optionally retry once with a lower intensity, without the event, or with a small `variation` and a new `seed`; otherwise skip the gesture (the dialogue goes on without it) |
| Mostly `velocity` / `acceleration` refusals | gesture still too fast after the slow-down (section 6), or `adapt=False` | report it to the dataset owner (calibration velocities, retraining) rather than raising limits in the JSON |
| `not_at_rest` | `from_rest=True` while the robot is elsewhere | call again with `from_rest=False` (or use `is_at_rest()` above) |
| `start_mismatch` | the robot moved since the measurement | measure again and regenerate |
| Uncalibrated file | `GuardianError` at load | export again from a calibrated project |
| Need to stop during the movement | — | run the movement asynchronously: `tid = motion.post.angleInterpolation(...)`, then `motion.stop(tid)` (generic NAOqi module method) or `motion.killTasksUsingResources(names)`. The project stops by **holding the posture**, not by going back to rest. Relaxing instead makes raised arms fall. |
| Connection lost to NAOqi during a blocking call | the call raises in your process; the robot finishes or stops on its own side | do not send a new movement before measuring the posture again |

What the project's controller adds on top, which an integration may want to copy: it streams the
movement at 25 Hz with `setAngles` (not one `angleInterpolation` call) to compare measured and
commanded angles, warns when a joint lags by more than 0.35 rad, and stops when the gap persists
for 0.5 s or exceeds 0.8 rad (`configs/robot.yaml` → `execution.tracking_*`). A single
`angleInterpolation` call has no such tracking check.

## 8. Checklist for the integrating agent

1. Copy `nao_gesture.py` and `nao_gesture.json` together; never edit the JSON limits by hand.
2. Load once at startup; log `gen.info` (calibration, model runs) and `gen.emotions`.
3. Map your own emotion vocabulary to `gen.emotions`; clamp intensity to [0, 1].
4. Express targets in the torso frame, azimuth positive to the robot's left, or pass
   `frame="head"` with the head angles measured at detection time.
5. Stiffen the gesture joints in place, then measure `current_pose` with sensor values.
6. Call `generate(..., current_pose=pose, from_rest=is_at_rest(pose), return_to=...)`.
7. Send only if `g.ok`, with `g.angle_interpolation()`; convert names to `str` under Python 2.7.
8. At the end: relax only at rest (or keep stiff at start to chain gestures); after a stop, hold.
9. Never command the legs; never bypass the guardian; never return to rest automatically after an
   error without a fresh measurement and a validated plan.
10. After any new calibration or training, re-export and replace both files.
