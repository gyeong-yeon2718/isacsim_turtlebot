"""Screen the control stack without Isaac Sim, and print the evidence.

Run this first, and run it after any gain change:

    python validate.py                    # default campaign
    python validate.py --repeats 60        # tighter statistics
    python validate.py --pairs 1-4 --trace # one pair, with a state trace

It checks, in order:

1. the transcribed link-efficiency model reproduces the poster's two anchors;
2. the geometry closes -- routes fit on the plywood, and the tags the fine
   alignment depends on are actually inside a camera's field of view at the
   docked pose;
3. the steering law's feasibility predicate agrees with the closed-form
   critically-damped solution it is supposed to integrate;
4. the closed-form registration recovers a known transform;
5. the full mission succeeds repeatedly, over randomised hand placement,
   calibration error, odometry bias, corner noise and dropout.

Exit status is non-zero if any check fails or the charging rate is below
``--min-rate``, so it is usable as a gate rather than only as a report.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from wpt_dock.apriltag import (
    TagDetectorSim,
    build_tags,
    nominal_cameras,
    perturbed_cameras,
    to_ground_correspondences,
    visible_tag_report,
)
from wpt_dock.config import DEFAULTS, Settings, strict
from wpt_dock.geometry import compose, rot2, wrap_angle
from wpt_dock.kinematic_sim import run_campaign, run_episode
from wpt_dock.path_follow import convergence_distance
from wpt_dock.registration import fit_rigid_2d, robot_pose_from_registration
from wpt_dock.routes import RouteBook, plan_route, point_is_on_board, route_fits_on_board

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------


def report_wpt(s: Settings) -> None:
    print("\n== link-efficiency model (transcribed from the project poster) ==")
    w = s.wpt
    print(f"  U = k*sqrt(Q_tx*Q_rx), Q_tx = Q_rx = {w.q_tx:g};  eta = U^2/(1+sqrt(1+U^2))^2")
    eta_aligned = w.efficiency(0.0)
    eta_mis = w.efficiency(w.misalign_offset)
    check(
        "k = 0.50 reproduces eta = 81.9 %",
        abs(eta_aligned - 0.819) < 0.002,
        f"got {eta_aligned * 100:.2f} %",
    )
    check(
        "k = 0.12 reproduces eta = 44.4 %",
        abs(eta_mis - 0.444) < 0.002,
        f"got {eta_mis * 100:.2f} % at the {w.misalign_offset * 1000:.0f} mm anchor",
    )
    print("   offset[mm]      k     eta[%]   eta/eta0[%]")
    for mm in (0, 2, 5, 8, 10, 15, 20, 25):
        d = mm / 1000.0
        print(f"     {mm:5d}   {w.coupling_k(d):6.3f}   {w.efficiency(d) * 100:6.2f}   "
              f"{w.relative_efficiency(d) * 100:8.2f}")
    print(f"  lock needs eta >= {w.lock_efficiency * 100:.0f} %; the poster's +-1 cm edge sits at "
          f"{w.efficiency(w.stable_offset) * 100:.1f} %")


def report_geometry(s: Settings) -> None:
    print("\n== board geometry ==")
    b = s.board
    ox, oy = b.stage_origin
    print(f"  stage {b.stage_size[0]:.3f} x {b.stage_size[1]:.3f} m, coil spacing "
          f"{b.coil_spacing[0]:.3f} x {b.coil_spacing[1]:.3f} m")
    print(f"  stage origin in coil-1 coordinates: ({ox:+.4f}, {oy:+.4f}) m")
    a, bb = s.robot.footprint_half_extents
    print(f"  robot footprint half-extents {a * 1000:.1f} x {bb * 1000:.1f} mm "
          f"(width set by the {s.robot.plate_size[1] * 1000:.0f} mm custom plate), "
          f"swept radius {s.robot.swept_radius * 1000:.1f} mm")

    all_fit = all(
        point_is_on_board(b, s.robot.swept_radius, x, y) for x, y in b.coil_positions.values()
    )
    check("every coil can be occupied, and turned on, without overhanging the plywood", all_fit)

    print("\n== routes ==")
    book = RouteBook(b, s.robot.swept_radius).build()
    print(book.report())
    ok = all(route_fits_on_board(b, s.robot.swept_radius, r)[0] for r in book.routes.values())
    check("all twelve coil-to-coil routes fit on the board", ok)


def report_visibility(s: Settings) -> None:
    """Are the tags the fine alignment depends on actually visible when docked?

    This is the check that would have saved the upstream project the most time.
    Their configs record only a device index and a resolution -- no mount pose -- so
    nothing in the repo can answer "can the camera see the tag at the moment of
    alignment?".  Here it is answered before anything else runs.
    """
    print("\n== tag visibility at the docked pose ==")
    tags = build_tags(s.board)
    rng = np.random.default_rng(7)
    cams = perturbed_cameras(s.cameras, s.detection, rng)
    detector = TagDetectorSim(cams, tags, s.detection, rng)

    worst = 99
    for coil in sorted(s.board.coil_positions):
        for heading_name, heading in (("+X", 0.0), ("+Y", math.pi / 2),
                                      ("-X", math.pi), ("-Y", -math.pi / 2)):
            pose = (*s.board.coil_positions[coil], heading)
            dets = detector.detect(pose)
            n_tags = len({d.tag_id for d in dets})
            corr = to_ground_correspondences(dets, nominal_cameras(s.cameras), tags, s.detection)
            worst = min(worst, n_tags)
            print(f"  coil {coil} facing {heading_name}: {n_tags} tag(s), {corr.n} corners"
                  f"  [{visible_tag_report(dets)}]")
    check("at least one tag is visible from every docked pose", worst >= 1,
          f"worst case {worst} tags")

    # And how good is a single-frame fix from that geometry?
    pose = (*s.board.coil_positions[4], math.pi / 2)
    errs = []
    for seed in range(200):
        r = np.random.default_rng(1000 + seed)
        cams_t = perturbed_cameras(s.cameras, s.detection, r)
        det = TagDetectorSim(cams_t, tags, s.detection, r)
        corr = to_ground_correspondences(det.detect(pose), nominal_cameras(s.cameras), tags, s.detection)
        if corr.n < 4:
            continue
        fit = fit_rigid_2d(corr.model, corr.observed, weights=corr.weights)
        if not fit.ok:
            continue
        est, _ = robot_pose_from_registration(fit)
        errs.append((est[0] - pose[0], est[1] - pose[1], wrap_angle(est[2] - pose[2])))
    if errs:
        a = np.abs(np.asarray(errs))
        print(f"  single-frame fix on coil 4 over {len(errs)} draws: "
              f"|dx| mean {a[:, 0].mean() * 1000:.2f} mm, |dy| mean {a[:, 1].mean() * 1000:.2f} mm, "
              f"|dyaw| mean {math.degrees(a[:, 2].mean()):.3f} deg")
        check("a single camera frame already resolves the pose to better than the tolerance",
              a[:, 0].mean() < s.dock.pos_tol and a[:, 1].mean() < s.dock.pos_tol)


def check_feasibility_math(s: Settings) -> None:
    """The integrated predicate must match the closed form it stands in for.

    For critical damping the arc-length solution is ``e(x) = (A + B x) exp(-lam x)``
    with ``A = e0`` and ``B = psi0 + lam*e0``.  If the RK4 integration in
    ``convergence_distance`` disagrees with that, the whole feasibility decision --
    and therefore every retreat -- is being made on a wrong number.
    """
    print("\n== feasibility predicate versus the closed form ==")
    g = s.docking
    lam = g.decay_rate
    tol = s.dock.pos_tol
    yaw_tol = s.dock.yaw_tol

    worst = 0.0
    for e0 in (0.005, 0.010, 0.020, 0.040):
        for psi0 in (0.0, 0.02, -0.05):
            a = e0
            b = psi0 + lam * e0

            def envelope(x: float) -> tuple[float, float]:
                ex = (a + b * x) * math.exp(-lam * x)
                # psi = de/dx
                psi = (b - lam * (a + b * x)) * math.exp(-lam * x)
                return ex, psi

            # Last distance at which either error is still out of tolerance.
            step = 0.004
            last_bad = -1
            n = int(4.0 / step)
            for i in range(n + 1):
                ex, psi = envelope(i * step)
                if abs(ex) > tol or abs(psi) > yaw_tol:
                    last_bad = i
            closed = 0.0 if last_bad < 0 else (last_bad + 1) * step
            got = convergence_distance(e0, psi0, g, tol, yaw_tol)
            worst = max(worst, abs(got - closed))
    check("RK4 predicate matches the analytic envelope", worst <= 0.010,
          f"worst disagreement {worst * 1000:.2f} mm")

    print("  required approach distance for a given starting lateral error "
          f"(lam = {lam:g}/m, zeta = {g.damping:g}):")
    for mm in (5, 10, 15, 20, 30, 40):
        d = convergence_distance(mm / 1000.0, 0.0, g, tol, yaw_tol)
        print(f"    {mm:3d} mm -> {d * 1000:6.1f} mm of run needed")


def check_registration() -> None:
    print("\n== closed-form registration ==")
    rng = np.random.default_rng(3)
    model = rng.uniform(-0.15, 0.15, size=(12, 2))
    truth = (0.031, -0.017, math.radians(4.5))
    observed = model @ rot2(truth[2]).T + np.array(truth[:2])
    fit = fit_rigid_2d(model, observed, sigma=1e-9)
    err = (
        abs(fit.pose[0] - truth[0]),
        abs(fit.pose[1] - truth[1]),
        abs(wrap_angle(fit.pose[2] - truth[2])),
    )
    check("exact data is recovered exactly", fit.ok and max(err) < 1e-9,
          f"max component error {max(err):.2e}")

    # Weighted fit: heavily down-weighted outliers must not drag the answer.
    noisy = observed.copy()
    noisy[0] += 0.05
    noisy[1] -= 0.04
    w = np.ones(len(model))
    w[0] = w[1] = 1e-6
    fit_w = fit_rigid_2d(model, noisy, weights=w)
    err_w = max(
        abs(fit_w.pose[0] - truth[0]),
        abs(fit_w.pose[1] - truth[1]),
        abs(wrap_angle(fit_w.pose[2] - truth[2])),
    )
    check("weights actually suppress bad points", fit_w.ok and err_w < 1e-4,
          f"max component error {err_w:.2e}")


def check_camera_roundtrip(s: Settings) -> None:
    print("\n== camera projection round trip ==")
    cams = nominal_cameras(s.cameras)
    rng = np.random.default_rng(11)
    worst = 0.0
    tested = 0
    for cam in cams.values():
        for _ in range(3000):
            p = np.array([rng.uniform(-0.1, 0.35), rng.uniform(-0.3, 0.3), 0.0])
            proj = cam.project(p)
            if proj is None:
                continue
            u, v, _ = proj
            if not cam.in_image(u, v):
                continue
            back = cam.backproject_to_ground(u, v)
            if back is None:
                continue
            worst = max(worst, float(np.linalg.norm(back - p[:2])))
            tested += 1
    check("project then back-project returns the same ground point", tested > 100 and worst < 1e-9,
          f"{tested} points, worst {worst:.2e} m")


def run_missions(s: Settings, pairs: list[tuple[int, int]], repeats: int, min_rate: float) -> None:
    print(f"\n== mission campaign: {len(pairs)} coil pair(s) x {repeats} randomised runs ==")
    camps = run_campaign(s, pairs, repeats)
    for key in [k for k in camps if k != "all"]:
        print(f"\n  -- {key} --")
        for line in camps[key].summary().splitlines():
            print("  " + line)
    print("\n  == aggregate ==")
    for line in camps["all"].summary().splitlines():
        print("  " + line)

    all_camp = camps["all"]
    rate = 100.0 * len(all_camp.successes) / max(1, all_camp.n)
    check(f"charging rate >= {min_rate:.0f} %", rate >= min_rate, f"got {rate:.1f} %")

    ok = all_camp.successes
    if ok:
        worst_r = max(r.radial_offset for r in ok)
        worst_yaw = max(abs(r.true_errors[2]) for r in ok)
        check(
            "every charging run is inside the stated tolerance in GROUND TRUTH, "
            "not merely in the robot's belief",
            worst_r <= math.sqrt(2) * s.dock.pos_tol + 1e-9 and worst_yaw <= s.dock.yaw_tol + 1e-9,
            f"worst radial {worst_r * 1000:.2f} mm, worst yaw {math.degrees(worst_yaw):.2f} deg",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # The defaults reproduce the campaign quoted in the README: all twelve ordered
    # coil pairs, 25 randomised runs each.  Takes a few minutes; use a smaller
    # --repeats while iterating on a gain.
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--pairs", default="1-2,2-1,1-3,3-1,1-4,4-1,2-3,3-2,2-4,4-2,3-4,4-3",
                    help="comma-separated start-target coil pairs, e.g. 1-4,2-3")
    ap.add_argument("--strict", action="store_true",
                    help="use the tighter preset: the poster's +-1 cm / 2 deg sweet spot")
    ap.add_argument("--min-rate", type=float, default=95.0)
    ap.add_argument("--trace", action="store_true", help="print a state trace for the first pair")
    args = ap.parse_args()

    s = strict() if args.strict else DEFAULTS
    pairs = []
    for token in args.pairs.split(","):
        a, b = token.strip().split("-")
        pairs.append((int(a), int(b)))

    print(f"wpt_dock validation  (tolerance {s.dock.pos_tol * 1000:.0f} mm / "
          f"{math.degrees(s.dock.yaw_tol):.1f} deg, dwell {s.dock.hold_time:.1f} s)")
    report_wpt(s)
    report_geometry(s)
    report_visibility(s)
    check_camera_roundtrip(s)
    check_feasibility_math(s)
    check_registration()
    run_missions(s, pairs, args.repeats, args.min_rate)

    if args.trace:
        start, target = pairs[0]
        print(f"\n== state trace, coil {start} -> coil {target} ==")
        r = run_episode(s, start, target, 20260811, collect_trace=True)
        last = None
        for row in r.trace:
            if row["state"] != last:
                ex, ey, eyaw = row["true_err"]
                print(f"  t={row['t']:6.2f}s  {row['state']:9s}  "
                      f"true err x={ex * 1000:+8.2f} y={ey * 1000:+8.2f} mm "
                      f"yaw={math.degrees(eyaw):+7.2f} deg  eta={row['eta'] * 100:5.1f}%")
                last = row["state"]
        print("  " + r.one_line())

    print("\n" + ("=" * 70))
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): " + "; ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
