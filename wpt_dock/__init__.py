"""Wireless-power-transfer coil alignment for a TurtleBot3, for Isaac Sim 5.1.

The package is split so that everything except the last two modules is pure
Python + NumPy and can be imported, tested and profiled without Isaac Sim:

    config          every tunable number, with its unit and its provenance
    geometry        SE(2) maths, angle bookkeeping, quaternion bridge
    routes          straight reference rays and coil-to-coil route planning
    path_follow     the Lyapunov steering law and the feasibility test
    apriltag        pinhole cameras, tag geometry, the detector seam
    registration    closed-form weighted 2D rigid fit, covariance, the EKF
    sensors         wheel-odometry error model
    estimator       odometry + tag registration fusion
    coupling        the poster's link-efficiency model and the lock decision
    fsm             the mission state machine
    kinematic_sim   a headless twin of the whole loop, for Monte Carlo screening

    isaac/          the only modules that import omni/pxr

Importing this package does not import Isaac Sim.  That is deliberate: the whole
control stack has to be runnable, and testable, without a GPU.
"""

__version__ = "0.1.0"
