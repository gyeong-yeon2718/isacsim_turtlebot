"""The only modules that import ``omni`` / ``pxr``.

Nothing in the parent package imports this one, so the whole control stack stays
testable without Isaac Sim.  Importing anything here before ``SimulationApp`` has
been constructed (in standalone mode) will fail -- ``pxr`` is not on ``sys.path``
until Kit's plugin system has booted.
"""
