"""Unit tests for everything that does not need Isaac Sim.

    python -m unittest discover -s tests -v

``unittest`` rather than pytest so this runs with a bare interpreter, including
Isaac Sim's bundled one.

These are property tests where a property exists, and analytic comparisons where a
closed form exists.  The conventions being pinned down -- left-positive lateral
offset, scalar-first quaternions, degrees-versus-radians at the USD boundary -- are
exactly the ones that produce silently-wrong motion rather than an exception, so
they get explicit tests instead of trust.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wpt_dock.apriltag import (  # noqa: E402
    TagDetectorSim,
    TagModel,
    build_tags,
    nominal_cameras,
    perturbed_cameras,
    to_ground_correspondences,
)
from wpt_dock.config import DEFAULTS, BoardSpec, FollowGains, WptLinkModel  # noqa: E402
from wpt_dock.coupling import LinkMonitor  # noqa: E402
from wpt_dock.geometry import (  # noqa: E402
    compose,
    invert,
    invert_jacobian,
    quat_wxyz_from_yaw,
    rot2,
    sinc_unnormalised,
    wrap_angle,
    yaw_from_quat_wxyz,
)
from wpt_dock.kinematic_sim import integrate  # noqa: E402
from wpt_dock.path_follow import convergence_distance, follow, saturate  # noqa: E402
from wpt_dock.registration import Ekf2D, fit_rigid_2d, robot_pose_from_registration  # noqa: E402
from wpt_dock.routes import (  # noqa: E402
    Leg,
    RayReference,
    footprint_is_on_board,
    plan_route,
    point_is_on_board,
    retreat_room,
)


class TestAngles(unittest.TestCase):
    def test_wrap_range(self):
        for a in np.linspace(-20.0, 20.0, 401):
            w = wrap_angle(float(a))
            self.assertGreater(w, -math.pi - 1e-12)
            self.assertLessEqual(w, math.pi + 1e-12)
            self.assertAlmostEqual(math.sin(w), math.sin(float(a)), places=12)
            self.assertAlmostEqual(math.cos(w), math.cos(float(a)), places=12)

    def test_wrap_pi_edges_agree(self):
        # wrap(pi) and wrap(-pi) must not straddle the branch cut differently, or a
        # heading error of exactly 180 degrees flips sign between frames and the
        # robot oscillates.
        self.assertAlmostEqual(wrap_angle(math.pi), math.pi, places=12)
        self.assertAlmostEqual(wrap_angle(-math.pi), math.pi, places=12)

    def test_sinc_is_smooth_through_zero(self):
        self.assertAlmostEqual(sinc_unnormalised(0.0), 1.0, places=15)
        for x in (1e-9, 1e-6, 1e-4, 1e-3, 0.1, 1.0, 2.5):
            self.assertAlmostEqual(sinc_unnormalised(x), math.sin(x) / x, places=10)
            self.assertAlmostEqual(sinc_unnormalised(-x), math.sin(x) / x, places=10)


class TestSE2(unittest.TestCase):
    def test_compose_invert_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            a = (float(rng.normal()), float(rng.normal()), float(rng.uniform(-3, 3)))
            ident = compose(a, invert(a))
            self.assertAlmostEqual(ident[0], 0.0, places=12)
            self.assertAlmostEqual(ident[1], 0.0, places=12)
            self.assertAlmostEqual(wrap_angle(ident[2]), 0.0, places=12)

    def test_invert_jacobian_matches_finite_difference(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            p = np.array([rng.normal(), rng.normal(), rng.uniform(-3, 3)])
            analytic = invert_jacobian(tuple(p))
            numeric = np.zeros((3, 3))
            h = 1e-7
            for j in range(3):
                pp, pm = p.copy(), p.copy()
                pp[j] += h
                pm[j] -= h
                a = np.array(invert(tuple(pp)))
                b = np.array(invert(tuple(pm)))
                d = (a - b) / (2 * h)
                d[2] = wrap_angle(float(a[2] - b[2])) / (2 * h)
                numeric[:, j] = d
            self.assertLess(float(np.abs(analytic - numeric).max()), 1e-5)

    def test_quaternion_is_scalar_first(self):
        # Isaac Sim returns (w, x, y, z).  A silent xyzw assumption yields a yaw that
        # is right at 0 and wrong everywhere else, so check a non-trivial angle.
        for yaw in (0.0, 0.3, -1.2, 2.9, math.pi):
            q = quat_wxyz_from_yaw(yaw)
            self.assertAlmostEqual(wrap_angle(yaw_from_quat_wxyz(q) - yaw), 0.0, places=12)
        self.assertAlmostEqual(yaw_from_quat_wxyz((1.0, 0.0, 0.0, 0.0)), 0.0, places=12)
        # Unnormalised input must still work: physics queries sometimes return them.
        self.assertAlmostEqual(yaw_from_quat_wxyz((2.0, 0.0, 0.0, 0.0)), 0.0, places=12)


class TestReference(unittest.TestCase):
    def test_lateral_is_positive_to_the_left(self):
        ray = RayReference((0.0, 0.0), 0.0)          # pointing +X
        self.assertGreater(ray.project(0.0, 0.1).lateral, 0.0)   # +Y is left of +X
        self.assertLess(ray.project(0.0, -0.1).lateral, 0.0)

    def test_s_is_the_signed_longitudinal_error(self):
        ray = RayReference((0.5, 0.0), 0.0)
        self.assertAlmostEqual(ray.project(0.2, 0.0).s, -0.3, places=12)
        self.assertAlmostEqual(ray.project(0.7, 0.0).s, +0.2, places=12)

    def test_lateral_sign_matches_error_dynamics(self):
        # The controller relies on d(lateral)/dt = v*sin(psi).  Verify by integrating.
        ray = RayReference((1.0, 0.0), 0.0)
        pose = (0.0, 0.0, 0.2)                       # heading left of the ray
        e0 = ray.project(pose[0], pose[1]).lateral
        pose = integrate(pose, 0.05, 0.0, 0.1)
        e1 = ray.project(pose[0], pose[1]).lateral
        self.assertGreater(e1 - e0, 0.0)
        self.assertAlmostEqual(e1 - e0, 0.05 * math.sin(0.2) * 0.1, places=9)

    def test_reversed_ray_flips_the_tangent(self):
        ray = RayReference((0.0, 0.0), 0.4)
        self.assertAlmostEqual(wrap_angle(ray.reversed().heading - (0.4 + math.pi)), 0.0, places=12)


class TestSaturation(unittest.TestCase):
    def test_curvature_is_preserved(self):
        robot = DEFAULTS.robot
        for v, w in ((0.5, 3.0), (0.07, 2.5), (0.2, -4.0), (0.02, 0.9)):
            vs, ws, scale = saturate(v, w, robot, 0.35)
            self.assertLessEqual(scale, 1.0 + 1e-12)
            self.assertAlmostEqual(ws / vs, w / v, places=9,
                                   msg="scaling v and w together must not change the arc")

    def test_limits_are_respected(self):
        robot = DEFAULTS.robot
        vs, ws, _ = saturate(5.0, 9.0, robot, 0.35)
        self.assertLessEqual(abs(ws), 0.35 + 1e-9)
        left, right = robot.body_to_wheels(vs, ws)
        self.assertLessEqual(max(abs(left), abs(right)), robot.max_wheel_rate + 1e-9)

    def test_max_wheel_rate_supports_both_datasheet_limits(self):
        robot = DEFAULTS.robot
        straight, _ = robot.body_to_wheels(robot.max_body_speed, 0.0)
        _, spin = robot.body_to_wheels(0.0, robot.max_body_yaw_rate)
        self.assertLessEqual(abs(straight), robot.max_wheel_rate + 1e-9)
        self.assertLessEqual(abs(spin), robot.max_wheel_rate + 1e-9)

    def test_wheel_roundtrip(self):
        robot = DEFAULTS.robot
        for v, w in ((0.07, 0.3), (-0.02, -1.1), (0.0, 0.8)):
            v2, w2 = robot.wheels_to_body(*robot.body_to_wheels(v, w))
            self.assertAlmostEqual(v, v2, places=12)
            self.assertAlmostEqual(w, w2, places=12)


class TestConvergence(unittest.TestCase):
    def test_matches_critically_damped_closed_form(self):
        g = FollowGains(decay_rate=18.0, damping=1.0)
        lam = g.decay_rate
        tol, yaw_tol = 0.010, math.radians(2.0)
        for e0 in (0.005, 0.012, 0.025, 0.040):
            a, b = e0, lam * e0
            step, last_bad = 0.004, -1
            for i in range(int(4.0 / step) + 1):
                x = i * step
                ex = (a + b * x) * math.exp(-lam * x)
                psi = (b - lam * (a + b * x)) * math.exp(-lam * x)
                if abs(ex) > tol or abs(psi) > yaw_tol:
                    last_bad = i
            closed = 0.0 if last_bad < 0 else (last_bad + 1) * step
            got = convergence_distance(e0, 0.0, g, tol, yaw_tol)
            self.assertLess(abs(got - closed), 0.010, msg=f"e0={e0}")

    def test_monotone_in_initial_error(self):
        g = DEFAULTS.docking
        tol, yaw_tol = DEFAULTS.dock.pos_tol, DEFAULTS.dock.yaw_tol
        prev = -1.0
        for mm in (10, 15, 20, 25, 30, 40, 60):
            d = convergence_distance(mm / 1000.0, 0.0, g, tol, yaw_tol)
            self.assertGreaterEqual(d, prev)
            prev = d

    def test_zero_when_already_inside(self):
        g = DEFAULTS.docking
        self.assertEqual(
            convergence_distance(0.0, 0.0, g, DEFAULTS.dock.pos_tol, DEFAULTS.dock.yaw_tol), 0.0
        )

    def test_gain_increase_shortens_the_requirement(self):
        tol, yaw_tol = DEFAULTS.dock.pos_tol, DEFAULTS.dock.yaw_tol
        slow = convergence_distance(0.025, 0.0, FollowGains(decay_rate=12.0), tol, yaw_tol)
        fast = convergence_distance(0.025, 0.0, FollowGains(decay_rate=18.0), tol, yaw_tol)
        self.assertLess(fast, slow)


class TestFollower(unittest.TestCase):
    def test_lateral_error_decays_at_the_designed_spatial_rate(self):
        """The whole point of the gain parameterisation, checked end to end.

        Gains are chosen so the cross-track error decays as exp(-lam * distance),
        independent of speed.  Roll the real ``follow`` law forward at two different
        speeds and confirm both land on the same error after the same *distance*.
        """
        s = DEFAULTS
        ray = RayReference((2.0, 0.0), 0.0)
        results = []
        for cap in (0.015, 0.060):
            pose = (0.0, 0.020, 0.0)
            travelled = 0.0
            dt = 1.0 / 200.0
            while travelled < 0.25:
                res = follow(
                    ray, pose, gains=s.docking, speeds=s.speeds, robot=s.robot,
                    speed_cap=cap, stop_at=0.0, arrive_tol=0.001,
                )
                pose = integrate(pose, res.v, res.w, dt)
                travelled += abs(res.v) * dt
            results.append(ray.project(pose[0], pose[1]).lateral)
        self.assertLess(abs(results[0]), 0.004)
        self.assertLess(abs(results[1]), 0.004)
        # Same distance travelled -> comparable residual, regardless of speed.
        self.assertLess(abs(results[0] - results[1]), 0.0025)

    def test_spins_instead_of_arcing_when_badly_misaligned(self):
        s = DEFAULTS
        ray = RayReference((1.0, 0.0), 0.0)
        res = follow(ray, (0.0, 0.0, math.pi / 2), gains=s.cruise, speeds=s.speeds,
                     robot=s.robot, stop_at=0.0)
        self.assertEqual(res.mode, "spin")
        self.assertAlmostEqual(res.v, 0.0, places=12)
        self.assertLess(res.w, 0.0)          # turn clockwise, back towards +X

    def test_reverse_moves_backwards(self):
        s = DEFAULTS
        ray = RayReference((0.0, 0.0), 0.0)
        res = follow(ray.reversed(), (0.10, 0.0, 0.0), gains=s.docking, speeds=s.speeds,
                     robot=s.robot, reverse=True, speed_cap=0.03, stop_at=0.30)
        self.assertLess(res.v, 0.0)


class TestRegistration(unittest.TestCase):
    def test_recovers_an_exact_transform(self):
        rng = np.random.default_rng(3)
        model = rng.uniform(-0.2, 0.2, size=(10, 2))
        truth = (0.021, -0.013, math.radians(3.3))
        observed = model @ rot2(truth[2]).T + np.array(truth[:2])
        fit = fit_rigid_2d(model, observed, sigma=1e-9)
        self.assertTrue(fit.ok)
        self.assertAlmostEqual(fit.pose[0], truth[0], places=12)
        self.assertAlmostEqual(fit.pose[1], truth[1], places=12)
        self.assertAlmostEqual(wrap_angle(fit.pose[2] - truth[2]), 0.0, places=12)

    def test_never_produces_a_reflection(self):
        # A single atan2 cannot yield a reflection, unlike an unguarded SVD.  Confirm
        # the rotation stays proper even for deliberately mirrored data.
        model = np.array([[0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1]])
        mirrored = model.copy()
        mirrored[:, 1] *= -1.0
        fit = fit_rigid_2d(model, mirrored, sigma=0.01, max_residual=1.0)
        r = rot2(fit.pose[2])
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=12)

    def test_more_points_means_a_tighter_covariance(self):
        rng = np.random.default_rng(4)
        big = rng.uniform(-0.2, 0.2, size=(40, 2))
        few = big[:4]
        for pts in (few, big):
            obs = pts + rng.normal(0.0, 0.001, size=pts.shape)
            fit = fit_rigid_2d(pts, obs, sigma=0.001)
            self.assertTrue(fit.ok)
        f_few = fit_rigid_2d(few, few + rng.normal(0, 1e-4, few.shape), sigma=0.001)
        f_big = fit_rigid_2d(big, big + rng.normal(0, 1e-4, big.shape), sigma=0.001)
        self.assertLess(f_big.covariance[0, 0], f_few.covariance[0, 0])

    def test_weights_suppress_outliers(self):
        rng = np.random.default_rng(5)
        model = rng.uniform(-0.2, 0.2, size=(12, 2))
        truth = (0.01, 0.02, 0.05)
        observed = model @ rot2(truth[2]).T + np.array(truth[:2])
        observed[0] += 0.08
        w = np.ones(len(model))
        w[0] = 1e-8
        fit = fit_rigid_2d(model, observed, weights=w)
        self.assertTrue(fit.ok)
        self.assertLess(abs(fit.pose[0] - truth[0]), 1e-4)
        self.assertLess(abs(wrap_angle(fit.pose[2] - truth[2])), 1e-4)

    def test_robot_pose_inverts_the_fit(self):
        # fit.pose is the model frame seen from the robot; the robot pose is its
        # inverse.  Getting this backwards is a sign error that only shows up as the
        # robot driving away from the target.
        model = np.array([[0.1, 0.05], [0.1, -0.05], [0.24, 0.0], [0.24, 0.18]])
        true_robot = (0.4, -0.2, 0.7)
        observed = np.array(
            [list(compose(invert(true_robot), (float(p[0]), float(p[1]), 0.0))[:2]) for p in model]
        )
        fit = fit_rigid_2d(model, observed, sigma=1e-9)
        pose, cov = robot_pose_from_registration(fit)
        self.assertAlmostEqual(pose[0], true_robot[0], places=10)
        self.assertAlmostEqual(pose[1], true_robot[1], places=10)
        self.assertAlmostEqual(wrap_angle(pose[2] - true_robot[2]), 0.0, places=10)
        self.assertEqual(cov.shape, (3, 3))
        self.assertLess(float(np.abs(cov - cov.T).max()), 1e-15)


class TestEkf(unittest.TestCase):
    def _predict(self, ekf, v, w, dt):
        ekf.predict(v, w, dt, speed_noise=0.04, yaw_noise=0.05, yaw_from_speed=0.03,
                    floor_pos=8e-5, floor_yaw=2.5e-4)

    def test_straight_line_prediction_is_exact(self):
        ekf = Ekf2D((0.0, 0.0, 0.0))
        for _ in range(100):
            self._predict(ekf, 0.05, 0.0, 0.01)
        self.assertAlmostEqual(ekf.pose[0], 0.05, places=9)
        self.assertAlmostEqual(ekf.pose[1], 0.0, places=12)

    def test_pure_rotation_does_not_translate(self):
        ekf = Ekf2D((0.3, -0.2, 0.0))
        for _ in range(200):
            self._predict(ekf, 0.0, 0.8, 0.005)
        self.assertAlmostEqual(ekf.pose[0], 0.3, places=12)
        self.assertAlmostEqual(ekf.pose[1], -0.2, places=12)
        self.assertAlmostEqual(wrap_angle(ekf.pose[2] - 0.8), 0.0, places=9)

    def test_arc_prediction_matches_the_integrator(self):
        ekf = Ekf2D((0.0, 0.0, 0.0))
        pose = (0.0, 0.0, 0.0)
        for _ in range(120):
            self._predict(ekf, 0.06, 0.9, 1.0 / 120.0)
            pose = integrate(pose, 0.06, 0.9, 1.0 / 120.0)
        for a, b in zip(ekf.pose, pose):
            self.assertAlmostEqual(a, b, places=9)

    def test_covariance_grows_while_driving_blind(self):
        ekf = Ekf2D((0.0, 0.0, 0.0), 0.001, 0.001)
        before = ekf.position_sigma
        for _ in range(300):
            self._predict(ekf, 0.07, 0.0, 1.0 / 60.0)
        self.assertGreater(ekf.position_sigma, before)

    def test_floor_stops_overconfidence_and_keeps_p_positive_definite(self):
        """The fix for the divergence the Monte Carlo found.

        Repeated absolute fixes share a fixed calibration error, so the filter must
        not average its way below that error -- an overconfident filter starts
        rejecting the very measurements that would correct it.
        """
        floor_pos, floor_yaw = 0.0015, math.radians(0.5)
        ekf = Ekf2D((0.0, 0.0, 0.0), 0.05, 0.05,
                    floor_position_sigma=floor_pos, floor_yaw_sigma=floor_yaw)
        r = np.diag([1e-8, 1e-8, 1e-8])
        for _ in range(400):
            self.assertTrue(ekf.update_pose((0.0, 0.0, 0.0), r))
        self.assertGreaterEqual(ekf.position_sigma, floor_pos - 1e-12)
        self.assertGreaterEqual(ekf.yaw_sigma, floor_yaw - 1e-15)
        eigs = np.linalg.eigvalsh(ekf.p)
        self.assertGreater(float(eigs.min()), 0.0)
        self.assertLess(float(np.abs(ekf.p - ekf.p.T).max()), 1e-18)

    def test_gate_rejects_a_wild_fix(self):
        ekf = Ekf2D((0.0, 0.0, 0.0), 0.002, 0.002)
        self.assertFalse(ekf.update_pose((5.0, -3.0, 2.0), np.diag([1e-6, 1e-6, 1e-6])))

    def test_persistent_rejection_inflates_and_recovers(self):
        """A gate that keeps firing is evidence against the filter, not the sensor."""
        ekf = Ekf2D((0.0, 0.0, 0.0), 1e-4, 1e-4)
        r = np.diag([1e-8, 1e-8, 1e-8])
        accepted = False
        for _ in range(60):
            if ekf.update_pose((0.05, 0.0, 0.0), r):
                accepted = True
                break
        self.assertTrue(accepted, "the filter never recovered from a wrong, confident state")


class TestRoutes(unittest.TestCase):
    def test_same_row_and_column_are_single_legs(self):
        b = BoardSpec()
        for a, c in ((1, 2), (2, 1), (3, 4), (1, 3), (2, 4)):
            self.assertEqual(len(plan_route(b, a, c).legs), 1, msg=f"{a}->{c}")

    def test_diagonal_keeps_the_long_axis_for_the_final_approach(self):
        b = BoardSpec()
        long_axis = max(b.coil_spacing)
        for a, c in ((1, 4), (4, 1), (2, 3), (3, 2)):
            route = plan_route(b, a, c)
            self.assertEqual(len(route.legs), 2, msg=f"{a}->{c}")
            self.assertTrue(route.legs[-1].is_final)
            self.assertAlmostEqual(route.legs[-1].length, long_axis, places=9,
                                   msg="the final approach should get the longer axis")

    def test_leg_endpoints_line_up(self):
        b = BoardSpec()
        route = plan_route(b, 1, 4)
        start = route.legs[0].start_point()
        self.assertAlmostEqual(start[0], b.coil_positions[1][0], places=9)
        self.assertAlmostEqual(start[1], b.coil_positions[1][1], places=9)
        self.assertAlmostEqual(route.legs[-1].target[0], b.coil_positions[4][0], places=9)

    def test_footprint_check_is_orientation_aware(self):
        b = BoardSpec()
        half = DEFAULTS.robot.footprint_half_extents
        self.assertTrue(footprint_is_on_board(b, half, 0.0, 0.0, 0.0))
        # Far enough off the west edge that the support envelope really does leave the board.
        self.assertFalse(footprint_is_on_board(b, half, -0.120, 0.10, 0.0))
        # An asymmetric envelope would show a heading dependence here; a square one must not,
        # and asserting that keeps the two from being silently swapped.
        for yaw in (0.0, math.pi / 2, math.pi, 0.7):
            self.assertTrue(footprint_is_on_board(b, half, 0.0, 0.0, yaw))

    def test_containment_uses_the_support_envelope_not_the_overhang(self):
        """The 230 mm plate rides 156 mm up; it may overhang the plywood.

        If containment used the overall outline instead, coil 1 and coil 4 would both fall
        outside the safe area once the rear extension was added and every route would be
        rejected -- the robot being fine and the model of it being wrong.
        """
        b = BoardSpec()
        support = DEFAULTS.robot.footprint_half_extents
        overall = DEFAULTS.robot.overall_half_extents
        self.assertLess(support[0], overall[0])
        self.assertLess(support[1], overall[1])
        for coil, (x, y) in b.coil_positions.items():
            self.assertTrue(footprint_is_on_board(b, support, x, y, 0.0), msg=f"coil {coil}")
            self.assertTrue(
                point_is_on_board(b, DEFAULTS.robot.swept_radius, x, y), msg=f"turn at coil {coil}"
            )

    def test_obstacle_clearance_is_a_different_radius_from_board_containment(self):
        """Two radii, two questions, and conflating them put a conveyor through the robot.

        ``swept_radius`` answers "does the robot stay on the plywood" and is correctly built
        from the *support* envelope, because the overhanging plate has nothing to hit out there.
        A conveyor standing on the board does have something to hit, so obstacle clearance needs
        the low structure -- and the rear extension puts that 26 mm further out than a
        bounding-box half-extent would suggest, because the outline is not centred on the wheel
        axle it rotates about.
        """
        r = DEFAULTS.robot
        self.assertGreater(r.obstacle_swept_radius, r.swept_radius,
                           "obstacle clearance cannot be smaller than board containment here")

        # It must reach past the rear structure, which is the binding corner.
        rear_x = 0.5 * r.base_footprint[1] + r.rear_extension
        rear_corner = math.hypot(rear_x, 0.5 * r.base_footprint[0])
        self.assertAlmostEqual(r.obstacle_swept_radius, rear_corner, places=9,
                               msg="the rear extension corner should be the binding one")

        # hypot(overall_half_extents) is not this quantity, and the interesting part is that it
        # errs in *both* directions at once, so it cannot be patched into service:
        #   - it treats the outline as centred on the axle, which hides 26 mm of rear overhang,
        #   - and it folds in the 230 mm plate, which rides above a low obstacle entirely.
        # Here the second effect wins and it comes out larger, which would push the station
        # needlessly far away.  On a robot with a narrower plate the first would win and it
        # would put a conveyor through the rear extension.  Either way it is the wrong question.
        centred = math.hypot(*r.overall_half_extents)
        self.assertNotAlmostEqual(r.obstacle_swept_radius, centred, places=3)
        self.assertGreater(centred, r.obstacle_swept_radius,
                           "with a 230 mm plate the bounding box overstates the low outline")
        plate_corner = math.hypot(0.5 * r.plate_size[0], 0.5 * r.plate_size[1])
        self.assertLess(plate_corner, r.obstacle_swept_radius,
                        "the plate is not the binding structure, so excluding it changes nothing"
                        " about the answer -- only about the reasoning being right")

        # And board containment must still pass at every coil, unchanged by any of this.
        b = BoardSpec()
        for coil, (x, y) in b.coil_positions.items():
            self.assertTrue(point_is_on_board(b, r.swept_radius, x, y), msg=f"turn at coil {coil}")

    def test_retreat_room_leaves_an_edge_margin(self):
        b = BoardSpec()
        leg = plan_route(b, 2, 4).legs[-1]
        radius = DEFAULTS.robot.swept_radius
        room = retreat_room(b, radius, leg, 10.0, edge_margin=0.015)
        # The full distance from the coil to the safe-area edge, minus the margin.
        ox, oy = b.stage_origin
        full = leg.target[1] - (oy + radius)
        self.assertLess(room, full)
        self.assertGreater(room, full - 0.020)

    def test_retreat_room_never_negative(self):
        b = BoardSpec()
        leg = Leg(target=(0.0, 0.0), heading=0.0, length=0.1, name="x")
        self.assertGreaterEqual(retreat_room(b, 0.40, leg, 1.0), 0.0)


class TestWptModel(unittest.TestCase):
    def test_poster_anchors_reproduce(self):
        m = WptLinkModel()
        self.assertAlmostEqual(m.efficiency(0.0), 0.819, places=3)
        self.assertAlmostEqual(m.efficiency(m.misalign_offset), 0.444, places=3)

    def test_efficiency_is_monotone_decreasing(self):
        m = WptLinkModel()
        prev = 2.0
        for mm in range(0, 41):
            eta = m.efficiency(mm / 1000.0)
            self.assertLessEqual(eta, prev + 1e-12)
            prev = eta

    def test_relative_efficiency_starts_at_one(self):
        m = WptLinkModel()
        self.assertAlmostEqual(m.relative_efficiency(0.0), 1.0, places=12)

    def test_stated_one_centimetre_edge_clears_the_lock_threshold(self):
        m = WptLinkModel()
        self.assertGreaterEqual(m.efficiency(m.stable_offset), m.lock_efficiency - 1e-6)


class TestLockMonitor(unittest.TestCase):
    def test_dwell_is_required(self):
        s = DEFAULTS
        mon = LinkMonitor(s.dock, s.wpt)
        dt = 1.0 / 60.0
        elapsed = 0.0
        while elapsed < s.dock.hold_time - 2 * dt:
            state = mon.update(dt, 0.0, 0.0, 0.0)
            self.assertFalse(state.locked, "locked before the dwell completed")
            elapsed += dt
        for _ in range(4):
            state = mon.update(dt, 0.0, 0.0, 0.0)
        self.assertTrue(state.locked)

    def test_leaving_tolerance_resets_the_dwell(self):
        s = DEFAULTS
        mon = LinkMonitor(s.dock, s.wpt)
        for _ in range(30):
            mon.update(1.0 / 60.0, 0.0, 0.0, 0.0)
        self.assertGreater(mon.held, 0.0)
        mon.update(1.0 / 60.0, 0.05, 0.0, 0.0)
        self.assertEqual(mon.held, 0.0)

    def test_single_axis_violation_blocks_the_lock(self):
        s = DEFAULTS
        mon = LinkMonitor(s.dock, s.wpt)
        for _ in range(120):
            state = mon.update(1.0 / 60.0, 0.0, 0.0, math.radians(5.0))
        self.assertFalse(state.locked)
        self.assertEqual(state.worst_axis, "yaw")


class TestCameras(unittest.TestCase):
    def test_project_backproject_roundtrip(self):
        cams = nominal_cameras(DEFAULTS.cameras)
        rng = np.random.default_rng(9)
        tested = 0
        for cam in cams.values():
            for _ in range(2000):
                p = np.array([rng.uniform(-0.1, 0.35), rng.uniform(-0.3, 0.3), 0.0])
                proj = cam.project(p)
                if proj is None:
                    continue
                u, v, _ = proj
                if not cam.in_image(u, v):
                    continue
                back = cam.backproject_to_ground(u, v)
                self.assertIsNotNone(back)
                self.assertLess(float(np.linalg.norm(back - p[:2])), 1e-9)
                tested += 1
        self.assertGreater(tested, 100)

    def test_points_behind_the_camera_are_rejected(self):
        cam = nominal_cameras(DEFAULTS.cameras)["front"]
        self.assertIsNone(cam.project(np.array([0.0, 0.0, 5.0])))

    def test_nadir_ground_sigma_is_uniform_across_the_image(self):
        """For a straight-down camera the ground scale is exactly uniform.

        This started as an assertion that off-axis corners are noisier, which failed --
        and the failure was the *test* being wrong, which is worth recording.  A nadir
        pinhole looks at a plane parallel to its own image plane, so the mapping is a pure
        scaling: the ray direction's z component is constant, the ray parameter is
        constant, and ``d(ground)/d(pixel)`` is ``height / focal`` everywhere.  The
        ``1/cos^3`` falloff people remember belongs to *oblique* views.

        Consequence worth being honest about: with all three cameras nadir at the same
        height, the per-point weighting in ``fit_rigid_2d`` is a no-op for this
        configuration.  It is kept because it stops being a no-op the moment a camera is
        tilted or mounted at a different height -- see the next test.
        """
        for cam in nominal_cameras(DEFAULTS.cameras).values():
            centre = cam.ground_point_sigma(cam.spec.cx, cam.spec.cy, 0.35)
            corner = cam.ground_point_sigma(cam.spec.width * 0.03, cam.spec.height * 0.03, 0.35)
            expected = 0.35 * cam.spec.position[2] / cam.spec.fx
            self.assertAlmostEqual(centre, corner, places=12, msg=cam.name)
            self.assertAlmostEqual(centre, expected, places=9, msg=cam.name)

    def test_tilted_camera_ground_sigma_grows_with_obliquity(self):
        """A tilted camera *does* have a gradient, and the weighting must see it."""
        from wpt_dock.apriltag import CameraModel
        from wpt_dock.config import CameraSpec as Spec

        tilted = CameraModel(
            Spec("tilted", 640, 480, math.radians(62.0), (0.03, 0.0, 0.1545), 0.0,
                 math.radians(-45.0))
        )
        near = tilted.ground_point_sigma(tilted.spec.cx, tilted.spec.height * 0.85, 0.35)
        far = tilted.ground_point_sigma(tilted.spec.cx, tilted.spec.height * 0.15, 0.35)
        self.assertGreater(far, near, "the far end of an oblique view must be noisier")
        self.assertGreater(far / near, 1.5, f"gradient too weak: {far / near:.3f}")

    def test_tag_corners_rotate_with_the_tag(self):
        tag = TagModel(11, (0.5, 0.25), math.pi / 2, 0.03)
        corners = tag.corners_board()
        self.assertEqual(corners.shape, (4, 2))
        centre = corners.mean(axis=0)
        self.assertAlmostEqual(float(centre[0]), 0.5, places=12)
        self.assertAlmostEqual(float(centre[1]), 0.25, places=12)
        # A 90 degree tag maps its local +X corner offsets onto +Y.
        self.assertAlmostEqual(float(corners[1, 1] - corners[0, 1]), 0.03, places=12)

    def test_tag_ids_follow_the_upstream_scheme(self):
        tags = build_tags(BoardSpec())
        self.assertEqual(sorted(tags), sorted(
            [shelf * 10 + pos for shelf in (1, 2, 3, 4) for pos in (1, 2, 3, 4)]
        ))
        self.assertEqual(tags[41].shelf, 4)
        self.assertEqual(tags[41].position_name, "north")
        self.assertEqual(tags[34].position_name, "east")

    def test_docked_pose_yields_a_usable_fix(self):
        s = DEFAULTS
        tags = build_tags(s.board)
        rng = np.random.default_rng(21)
        cams_true = perturbed_cameras(s.cameras, s.detection, rng)
        detector = TagDetectorSim(cams_true, tags, s.detection, rng)
        for coil in (1, 2, 3, 4):
            for heading in (0.0, math.pi / 2, math.pi, -math.pi / 2):
                pose = (*s.board.coil_positions[coil], heading)
                corr = to_ground_correspondences(
                    detector.detect(pose), nominal_cameras(s.cameras), tags, s.detection
                )
                self.assertGreaterEqual(corr.n, 4, msg=f"coil {coil} heading {heading}")
                fit = fit_rigid_2d(corr.model, corr.observed, weights=corr.weights)
                self.assertTrue(fit.ok)
                est, _ = robot_pose_from_registration(fit)
                self.assertLess(math.hypot(est[0] - pose[0], est[1] - pose[1]), s.dock.pos_tol)
                self.assertLess(abs(wrap_angle(est[2] - pose[2])), s.dock.yaw_tol)


class TestArm(unittest.TestCase):
    """The cute_arm kinematics, checked against the one pose upstream documents."""

    def setUp(self):
        from wpt_dock.arm import ArmSpec

        self.spec = ArmSpec()

    def test_home_pose_matches_the_documented_rest_position(self):
        """Upstream documents the rest pose as (12, 0, 12) cm from the shoulder pivot.

        That single published number is the only external check available on this
        convention, so it is worth asserting rather than eyeballing: if the joint sign
        convention were flipped, everything downstream would still look plausible and be
        mirrored.

        Upstream's 12 cm is to the **gripper mount** -- it is a link length, quoted before
        anything is bolted to the wrist.  ``forward_kinematics`` reports the tool centre point,
        so the horizontal figure here is ``l_fore + l_tool``.  Asserting against the raw 0.120
        would be asserting that the tool offset does not exist, which is the bug this
        distinction was introduced to fix.
        """
        from wpt_dock.arm import HOME, forward_kinematics

        tip = forward_kinematics(self.spec, HOME)
        self.assertAlmostEqual(tip[0], self.spec.distal, places=9)
        self.assertAlmostEqual(tip[1], 0.000, places=9)
        # The vertical figure is the one that matches exactly, because it is a link length with no
        # ambiguity about its endpoints: the upper arm is straight up at HOME.
        self.assertAlmostEqual(tip[2], self.spec.l_upper, places=9)

        # The horizontal figure does *not* come out at 12.0 cm, and that is deliberate.  Upstream
        # publishes both the rest position and LENGTH_ELBOW_GRIPPER as 12 cm without saying which
        # point on the gripper they measure to, and the user resolved it: the point where the
        # gripper applies force is inboard of 120 mm.  So 12 cm is the bound this stays under.
        self.assertLess(tip[0], 0.120)
        self.assertGreater(tip[0], 0.100, "and it should not be far under it either")

    def test_ik_inverts_fk_over_the_workspace(self):
        from wpt_dock.arm import forward_kinematics, solve_ik

        rng = np.random.default_rng(5)
        tested = 0
        for _ in range(4000):
            target = (
                float(rng.uniform(0.04, 0.22)),
                float(rng.uniform(-0.12, 0.12)),
                float(rng.uniform(-0.12, 0.20)),
            )
            res = solve_ik(self.spec, target)
            if not res.ok:
                continue
            back = forward_kinematics(self.spec, res.pose)
            for a, b in zip(back, target):
                self.assertAlmostEqual(a, b, places=9)
            tested += 1
        self.assertGreater(tested, 500, "workspace sampling found too few reachable targets")

    def test_reach_limits_are_enforced_not_clamped(self):
        """Out-of-reach targets must be refused, not silently approximated.

        A solver that clamps returns a pose that looks fine and puts the gripper somewhere
        else, which in a pick-and-place run means dropping the payload next to the target
        rather than on it.
        """
        from wpt_dock.arm import solve_ik

        far = solve_ik(self.spec, (0.30, 0.0, 0.0))
        self.assertFalse(far.ok)
        self.assertIn("beyond", far.reason)
        near = solve_ik(self.spec, (0.01, 0.0, 0.0))
        self.assertFalse(near.ok)
        self.assertIn("dead zone", near.reason)
        # Just inside the documented envelope must succeed.
        self.assertTrue(solve_ik(self.spec, (0.225, 0.0, 0.0)).ok)

    def test_base_error_passes_straight_through_to_the_target(self):
        """10 mm of base error must be 10 mm of target error -- no more, no less.

        This is the link between the two halves of the project: the arm cannot see the
        shelf, so its placement accuracy *is* the alignment accuracy, and nothing in the
        transform is allowed to amplify or hide that.
        """
        from wpt_dock.arm import base_frame_target

        world = (0.30, 0.0, 0.10)
        exact = base_frame_target(self.spec, (0.0, 0.0, 0.0), 0.165, world)
        shifted = base_frame_target(self.spec, (0.010, 0.0, 0.0), 0.165, world)
        self.assertAlmostEqual(exact[0] - shifted[0], 0.010, places=12)
        lifted = base_frame_target(self.spec, (0.0, 0.0, 0.0), 0.175, world)
        self.assertAlmostEqual(exact[2] - lifted[2], 0.010, places=12)

    def test_rotating_the_base_rotates_the_target(self):
        """With the arm on the robot's centreline a 90 deg turn swaps the axes exactly.

        Checked with a zero mount offset on purpose.  With the real -15 mm offset the pivot
        itself moves when the robot turns, so the plain axis-swap identity does *not* hold --
        which is worth pinning down, because assuming it does is an easy way to introduce a
        15 mm error that only appears at non-zero headings.
        """
        from wpt_dock.arm import ArmSpec, base_frame_target

        centred = ArmSpec(mount_offset=(0.0, 0.0))
        world = (0.30, 0.0, 0.10)
        exact = base_frame_target(centred, (0.0, 0.0, 0.0), 0.165, world)
        turned = base_frame_target(centred, (0.0, 0.0, math.pi / 2), 0.165, world)
        self.assertAlmostEqual(turned[0], exact[1], places=9)
        self.assertAlmostEqual(turned[1], -exact[0], places=9)

        # And with the real offset the pivot translation must show up.
        offset = base_frame_target(self.spec, (0.0, 0.0, math.pi / 2), 0.165, world)
        self.assertAlmostEqual(offset[0], -self.spec.mount_offset[0], places=9)

    def test_sequencer_respects_the_servo_rate(self):
        from wpt_dock.arm import ArmSequencer, ArmPose, Waypoint

        seq = ArmSequencer(self.spec)
        target = ArmPose(math.radians(60.0), math.radians(90.0), math.radians(-90.0))
        seq.start([Waypoint("swing", joint_target=target, settle=0.0)])
        dt = 1.0 / 60.0
        steps = 0
        while not seq.finished and steps < 20000:
            before = seq.state.pose.base
            seq.step(dt)
            self.assertLessEqual(
                abs(seq.state.pose.base - before), self.spec.servo_rate * dt + 1e-12
            )
            steps += 1
        self.assertTrue(seq.succeeded)
        self.assertAlmostEqual(seq.state.pose.base, target.base, places=3)
        # 60 deg at 60 deg/s is about a second; assert the order of magnitude so an
        # accidental instantaneous jump would fail.
        self.assertGreater(steps * dt, 0.8)

    def test_carry_path_clears_the_robot_deck(self):
        """The lift-to-carry move must not drag the payload through the printed plate.

        This is the defect the user saw as "the object goes through the top plate".  Both
        endpoints of a joint-space move can clear the deck while the arc between them does
        not, so the whole interpolated path is checked -- and it is checked against the same
        plate height the build measures.  Going straight from the lift pose to CARRY leaves
        about -5 mm of clearance, which is why HOME is now in between.
        """
        from wpt_dock.arm import CARRY, HOME, PAYLOAD_HALF, ArmPose, path_clearance, solve_ik

        plate_top = 0.1653          # m, as measured by the Isaac build
        payload_half = PAYLOAD_HALF   # the scene's own constant, not a literal that can drift

        # The lift pose above the pick shelf, as the sequence actually solves it.
        lift = solve_ik(self.spec, (0.014, 0.135, -0.043))
        self.assertTrue(lift.ok, lift.reason)

        raised = ArmPose(lift.pose.base, HOME.shoulder, HOME.elbow)   # "raise in place"
        direct = path_clearance(self.spec, lift.pose, CARRY, plate_top, payload_half)
        staged = min(
            path_clearance(self.spec, lift.pose, raised, plate_top, payload_half),
            path_clearance(self.spec, raised, HOME, plate_top, payload_half),
            path_clearance(self.spec, HOME, CARRY, plate_top, payload_half),
        )
        self.assertGreater(staged, 0.010,
                           f"lift-then-slew must clear the deck, got {staged * 1000:.1f} mm")
        if math.isinf(direct) and math.isinf(staged):
            # Both infinite means neither path passes over the plate *at all*, so there is no
            # conflict left for staging to solve.  That is what the real gripper's geometry did:
            # once the tool centre point moved out to the jaw tips at 104 mm, the payload is
            # carried beyond the deck footprint the whole way.  Worth asserting as a fact rather
            # than forcing the old comparison, which would now be comparing two absences.
            self.assertGreater(self.spec.l_tool, 0.050,
                               "the conflict only disappears because the tool point is far out")
        else:
            self.assertGreater(staged, direct,
                               "staging has to beat the direct move or it is pointless")

    def test_kinematics_solve_for_the_tool_centre_point_not_the_wrist(self):
        """``forward_kinematics`` must reach the point the USD ``tcp`` prim marks.

        This pins down a defect worth naming: the tool offset existed as a literal in the USD
        chain and was missing from the kinematics, so the solver aimed the *wrist* at the shelf
        and the payload centre landed 16 mm further along the forearm.  With the forearm
        pointing nearly straight down at a shelf, almost all of that went into z, and every
        box came out 15.4 mm low -- buried in the shelf it was supposed to sit on.  The bias
        was constant, which is the tell: geometry, not noise.

        The check is that FK's distal reach equals ``l_fore + l_tool``, measured with the arm
        straight out, plus a round trip through the IK to show the two agree.
        """
        from wpt_dock.arm import ArmPose, forward_kinematics, solve_ik

        straight = forward_kinematics(self.spec, ArmPose(0.0, 0.0, 0.0))
        self.assertAlmostEqual(straight[0], self.spec.l_upper + self.spec.l_fore + self.spec.l_tool,
                               places=9)
        self.assertGreater(self.spec.l_tool, 0.0, "a gripper with no tool offset is a wrist")

        # Round trip: a target the place sequence actually uses, reached to sub-micron.
        for target in ((0.014, 0.135, -0.043), (0.150, 0.0, -0.050), (0.0, 0.170, 0.020)):
            res = solve_ik(self.spec, target)
            self.assertTrue(res.ok, f"{target}: {res.reason}")
            back = forward_kinematics(self.spec, res.pose)
            for got, want, axis in zip(back, target, "xyz"):
                self.assertAlmostEqual(got, want, places=9,
                                       msg=f"{target} {axis}: FK/IK disagree")

    def test_carry_pose_holds_the_payload_above_the_deck(self):
        from wpt_dock.arm import CARRY, HOME, PAYLOAD_HALF, payload_height

        plate_top = 0.1653
        for name, pose in (("HOME", HOME), ("CARRY", CARRY)):
            z = payload_height(self.spec, pose, plate_top)
            self.assertGreater(z - PAYLOAD_HALF, plate_top + 0.010,
                               f"{name} holds the payload too low: {z * 1000:.1f} mm")

    def test_the_payload_fits_between_the_jaws(self):
        """The carried object must be strictly inside the jaws' travel.

        The shipped value was a 35 mm cube against 34 mm of maximum jaw opening, and the pick
        sequence commanded the jaws to their 6 mm mechanical stop -- so they closed 14.5 mm
        into each side of the box.  The user's report was that the object was far too big for
        the gripper, and it was.  Two separate mistakes, so two separate assertions.
        """
        from wpt_dock.arm import PAYLOAD_SIZE

        # The *usable gap* is what an object passes through: pad centres minus pad thickness.
        open_gap = self.spec.gripper_clear(self.spec.gripper_open)
        closed_gap = self.spec.gripper_clear(self.spec.gripper_closed)
        self.assertLess(PAYLOAD_SIZE, open_gap,
                        "the jaws cannot open wide enough to go around the payload")
        self.assertGreater(PAYLOAD_SIZE, closed_gap,
                           "the payload is thinner than the closed jaws, so it cannot be gripped")
        self.assertGreater(0.5 * (open_gap - PAYLOAD_SIZE), 0.002,
                           "less than 2 mm of clearance per jaw is not a believable approach")

        # The pad *faces* must stop on the box, not at the mechanical limit and not sunk into
        # it -- the pads have thickness and the transforms place their centres.
        angle = self.spec.grip_angle_for(PAYLOAD_SIZE)
        self.assertAlmostEqual(self.spec.gripper_clear(angle), PAYLOAD_SIZE, places=9)
        self.assertAlmostEqual(self.spec.gripper_span(angle),
                               PAYLOAD_SIZE + self.spec.jaw_pad_thickness, places=9)
        self.assertGreater(angle, self.spec.gripper_closed,
                           "closing to the mechanical stop means the jaws pass through the box")
        self.assertLess(angle, self.spec.gripper_open, "the jaws have to actually move")

        # And an ungrippable object must be refused rather than mimed.
        with self.assertRaises(ValueError):
            self.spec.grip_angle_for(open_gap + 0.001)
        with self.assertRaises(ValueError):
            self.spec.grip_angle_for(closed_gap - 0.001)

    def test_the_fingers_pivot_and_never_turn_backwards(self):
        """The gripper is a scissor pair on vertical screws, so closing is a rotation.

        ``gripper.stl`` has exactly one bore, 1.90 mm across -- a 2 mm screw from the BOM, not an
        SG90's 4.8 mm spline -- so the fingers pivot on screws rather than sliding.  This pins the
        two properties that make the mapping usable: fully open is exactly zero rotation, and
        closing is monotone and positive.  Before ``jaw_pivot_offset`` was derived from
        ``span_open`` it was a free constant and "fully open" came out at -0.94 degrees.
        """
        s = self.spec
        # Closing the servo must close the jaw, monotonically, and never swing it inside the
        # fixed jaw.
        prev = 1e9
        for frac in (1.0, 0.75, 0.5, 0.25, 0.0):
            theta = s.jaw_rotation(frac * s.gripper_open)
            self.assertGreaterEqual(theta, 0.0, "the jaw must not swing past shut")
            self.assertLess(theta, prev, "closing the servo must close the jaw")
            prev = theta

        # Elbow to the force point must stay inside the documented 12 cm.  The documentation gives
        # that as an upper bound on the forearm; reading it as elbow-to-servo and then adding the
        # jaw's own 70 mm produced a 190 mm forearm, longer than the whole documented segment.
        self.assertLess(s.distal, 0.120,
                        "the gripper's force point is inboard of the documented 12 cm")
        self.assertAlmostEqual(s.distal, s.elbow_to_tool, places=12)
        self.assertGreater(s.l_fore, 0.0, "the forearm cannot be shorter than the jaw it carries")

        # The tool centre point is the tip contact, not a number near the wrist.  This is the
        # "force application point" the user kept reporting as wrong: the IK aims at l_tool, so if
        # it does not sit where the jaws actually meet, every grasp is off by the difference.
        self.assertAlmostEqual(s.l_tool, s.jaw_pivot_x + s.jaw_tip_reach, places=12)
        self.assertGreater(s.l_tool, 0.050,
                           "a tool point near the wrist cannot be where this gripper holds things")

        # The gap is purely the tip's lateral travel, because the tips meet when shut.
        for frac in (0.25, 0.5, 1.0):
            a = frac * s.gripper_open
            self.assertAlmostEqual(s.jaw_tip_reach * math.sin(s.jaw_rotation(a)),
                                   s.gripper_span(a), places=9)

    def test_the_measured_coil_is_the_source_before_transit(self):
        """``current_coil`` must name the coil ``coil_errors`` is measured against.

        The lights are driven from the link monitor, which is fed ``coil_errors``.  If the
        supervisor reports the *final* target while the active alignment is on the *source*,
        the source coil's full-coupling measurement gets attributed to the destination -- and
        the destination lights up, and steps to charging-green, from t = 0 with the robot half
        a metre away.  That is what the user saw.  This pins the invariant that prevented it.
        """
        from wpt_dock.arm import ArmSpec
        from wpt_dock.config import DEFAULTS
        from wpt_dock.fsm import MissionController
        from wpt_dock.pickplace import ALIGN_SOURCE, TRANSIT, PickPlaceMission

        board = DEFAULTS.board
        grasp = (board.coil_positions[1][0] - 0.10, board.coil_positions[1][1], 0.1175)
        drop = (board.coil_positions[4][0] + 0.10, board.coil_positions[4][1], 0.1175)
        m = PickPlaceMission(DEFAULTS, ArmSpec(), 1, 4,
                             grasp_world=grasp, drop_world=drop, plate_top_z=0.1653)

        self.assertEqual(m.phase, ALIGN_SOURCE)
        self.assertEqual(m.current_coil, 1,
                         "while aligning on the source, the measured coil is the source")
        self.assertNotEqual(m.current_coil, m.target_coil,
                            "if these were equal the test could not detect the defect")

        # Force the supervisor into the transit alignment and re-check.
        m.phase = TRANSIT
        m.mission = MissionController(DEFAULTS, 1, 4, initial_heading=0.0)
        self.assertEqual(m.current_coil, 4, "once driving to the target, that is the measured coil")

    def test_unreachable_waypoint_fails_the_sequence(self):
        from wpt_dock.arm import ArmSequencer, Waypoint

        seq = ArmSequencer(self.spec)
        seq.set_ik_context((0.0, 0.0, 0.0), 0.165)
        seq.start([Waypoint("impossible", world_target=(2.0, 0.0, 0.0))])
        self.assertTrue(seq.finished)
        self.assertFalse(seq.succeeded)
        self.assertIn("unreachable", seq.state.message)


class TestBoardGeometry(unittest.TestCase):
    def test_coil_array_is_centred_on_the_stage(self):
        b = BoardSpec()
        ox, oy = b.stage_origin
        self.assertAlmostEqual(ox + b.stage_size[0] / 2, b.coil_spacing[0] / 2, places=12)
        self.assertAlmostEqual(oy + b.stage_size[1] / 2, b.coil_spacing[1] / 2, places=12)

    def test_tags_sit_at_the_documented_distance(self):
        b = BoardSpec()
        positions = b.tag_positions()
        for shelf, (cx, cy) in b.coil_positions.items():
            for pos in (1, 2, 3, 4):
                tx, ty, _ = positions[shelf * 10 + pos]
                self.assertAlmostEqual(math.hypot(tx - cx, ty - cy), b.coil_center_to_tag, places=12)

    def test_every_coil_fits_with_the_swept_radius(self):
        b = BoardSpec()
        radius = DEFAULTS.robot.swept_radius
        ox, oy = b.stage_origin
        for x, y in b.coil_positions.values():
            self.assertGreaterEqual(x, ox + radius - 1e-9)
            self.assertLessEqual(x, ox + b.stage_size[0] - radius + 1e-9)
            self.assertGreaterEqual(y, oy + radius - 1e-9)
            self.assertLessEqual(y, oy + b.stage_size[1] - radius + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
