# The start pose inside the gesture's B-spline

How the exported generator (`nao_gesture.py`) starts and ends a gesture on the calibrated
**start pose**, inside the gesture's own cubic B-spline, instead of adding transitions around
it. API and segment list: [generator-integration.md](generator-integration.md).

## Before and after, one joint

Before: the learned B-spline started on its first learned point, away from the start pose.
Two transitions were added around it, each at least 0.4 s long (a fixed floor). The one at the
end was flat when the curve already ended on the start pose: 0.4 s of stillness.

```
angle
  ^                    learned B-spline
  |                   .--.      .-.
  |                  /    \    /   \
  |                 /      '--'     \
  |   entry       ./                 \
  |  transition  /                    \        flat transition (0.4 s)
S |-------------'                      '------------------------------
  +-------------+----------------------+---------------+--------------> t
     to_start    start_to_gesture        gesture         gesture_to_start
```

After: the start pose is part of the same B-spline, at both ends. The curve leaves it and
reaches it with a horizontal tangent (zero velocity and acceleration). Nothing is added.

```
angle
  ^               .--.      .-.
  |              /    \    /   \
  |             /      '--'     \
  |           ./                 \.
S |----------'                     '-----------
  +----------+---------------------+----------> t
     entry         learned part       return
  |<------------- gesture (one B-spline) ------>|
```

## Control points: `S S S P1 ... Pn S S S`

`S` is the start pose, `P1 ... Pn` the learned control points of the gesture.

```
 S   S   S   P1   P2   P3  ...  Pn   S   S   S
 \_______/                           \_______/
 start pose x3                     start pose x3
 v = 0, a = 0 at t = 0             v = 0, a = 0 at the end
```

Why three: a clamped cubic B-spline passes through its first control point, with a velocity
pointing to the second one and an acceleration set by the third.

```
 S  P1 ...       starts on S, but already moving (velocity jump)
 S  S  P1 ...    starts at zero velocity, acceleration jump
 S  S  S  P1 ... starts at zero velocity and zero acceleration
```

The same holds at the end. Repeating `S` is **not** a pause: there is no plateau. It is the
acceleration ramp (and, at the end, the braking ramp), counted in the movement.

## Several draws in one answer

With `duration_s`, several learned gestures are blended into one curve. The junction points
between draws are dropped, so the movement never goes through `S` in the middle:

```
 S S S  P1 ... Pn  Q1 ... Qm  S S S
        \______/   \______/
         draw 1     draw 2        (cut on a control point, then back to S)
```

## Between two answers

Two answers are two trajectories: answer N is played to its end, then N+1 is measured,
generated and sent. N ends at rest on `S`, N+1 starts at rest from `S`:

```
angle
  ^      answer N                       answer N+1
  |    .--.    .-.                     .-.     .--.
  |   /    \  /   \                   /   \   /    \
  |  /      ''     \.               ./     '-'      \
S |-'                '-------------'                 '---
  +-------------------+-----------+--------------------> t
                      |  short    |
                      |  stop     |  (measure + generate + send:
                                      no fixed transition any more)
```

Smooth braking, a brief stop on `S`, smooth restart.

## Segments, before and after

```
before:  to_start | start_to_gesture | gesture | gesture_to_start [| start_to_rest]
after:  [to_start]| gesture                                       [| start_to_rest]
```

What remains outside the curve:

- `to_start`: only when the measured posture is further than `start_tolerance_rad` from the
  start pose (e.g. a gesture interrupted earlier). At the start pose, the `gesture` segment
  is the whole movement.
- `rest_to_start` (`from_rest=True`) and `start_to_rest` (`return_to="rest"`): unchanged, still
  checked under the rest rules.

With `duration_s=D`, the `gesture` segment (entry and return included) lasts about `D`, so the
movement and the speech have the same length.

## Staying in the zone

A B-spline lies inside the convex hull of its control points. `S` is inside the guardian zone
and the learned points are projected into it, so the curve, entry and return included, stays
inside the zone for the position limits. Velocity and acceleration are still checked by the
guardian, and the usual playable adaptation slows down a moment that would go too fast.

## Entry and return time

The first learned point can be far from `S` (wrists turning by more than a radian, for
example). The time kept before it, the **entry**, is the longest of:

- `ENTRY_SPANS` (2) knot spans of the first draw;
- the minimum-jerk duration of the move `S -> P1` for each joint, within 0.8 x its velocity
  and acceleration limits: `max(1.875 d / v, sqrt(5.774 d / a))`.

The **return** after the last kept point uses the same rule (`RETURN_SPANS` knot spans, or the
minimum-jerk duration back to `S`). Both are reported in `g.sequence` (`entry_s`, `return_s`)
when `duration_s` or `count` is given.
