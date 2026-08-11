r"""Exercise the **GUI** branch of ``run_in_isaacsim.py`` from the terminal.

    C:\isaacsim\python.bat check_gui_path.py

Why this exists: the entry file has two code paths, and only the standalone one is
reachable from ``python.bat``.  The GUI path is the one the request actually cares
about, and it is also the riskier of the two -- it uses the ``*_async`` lifecycle
(``create_new_stage_async``, ``initialize_simulation_context_async``,
``reset_async``, ``play_async``), a physics callback instead of a loop, and a
timeline callback to survive STOP.  Shipping that unexecuted would be shipping a
guess.

How it fakes the GUI faithfully:

1. Boot Kit via ``SimulationApp``, which sets
   ``builtins.ISAAC_LAUNCHED_FROM_TERMINAL = False``.
2. Set that flag back to **True** *before* any ``isaacsim.core.api`` import.  This is
   the exact switch the GUI runs on: with it True, ``SimulationContext.__init__``
   skips its synchronous stage setup and leaves ``_physics_context`` as None, so the
   ``await initialize_simulation_context_async()`` in the GUI path becomes load
   bearing rather than decorative.
3. ``exec`` the entry file, which now takes the GUI branch and schedules
   ``_gui_main()`` on Kit's own async loop.
4. Pump ``simulation_app.update()`` -- which is what Kit's main loop does every frame --
   so the coroutine and the physics callbacks actually run.

What it does *not* cover: the Script Editor window itself, and the fact that its Run
button executes on the UI thread.  That is precisely why the GUI path uses a physics
callback and never a ``while`` loop, and this harness verifies that the callback
route works.
"""

from __future__ import annotations

import builtins
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

# Step 2: pretend we are inside the GUI.  Must happen before isaacsim.core.api loads.
builtins.ISAAC_LAUNCHED_FROM_TERMINAL = True

entry = os.path.join(PROJECT_DIR, "run_in_isaacsim.py")
source = open(entry, encoding="utf-8").read()

namespace: dict = {"__name__": "__wpt_gui_check__", "__file__": entry}
# Force a short, deterministic mission so the check finishes quickly.
source = source.replace("START_COIL = 1", "START_COIL = 2", 1)
source = source.replace("TARGET_COIL = 4", "TARGET_COIL = 4", 1)

print("[check] executing the entry file with the GUI branch forced", flush=True)
exec(compile(source, entry, "exec"), namespace)

runner = getattr(builtins, "WPT_DOCK_RUNNER", None)
if namespace.get("IN_GUI") is not True:
    print("[check] FAIL: the entry file did not take the GUI branch "
          f"(IN_GUI={namespace.get('IN_GUI')})", flush=True)
    simulation_app.close()
    sys.exit(1)

# Step 4: pump frames.  The coroutine has only been *scheduled* at this point; it
# makes progress on Kit's async loop, one step per app update.
MAX_FRAMES = 30000
frames = 0
while simulation_app.is_running() and frames < MAX_FRAMES:
    simulation_app.update()
    frames += 1
    if runner is None:
        runner = getattr(builtins, "WPT_DOCK_RUNNER", None)
    elif runner.finished:
        break

print("-" * 78, flush=True)
if runner is None:
    print(f"[check] FAIL: _gui_main never completed setup after {frames} frames", flush=True)
    code = 1
elif not runner.finished:
    print(f"[check] FAIL: the mission did not finish within {frames} frames; "
          f"last state was {runner.status.state if runner.status else 'none'}", flush=True)
    code = 1
else:
    status = runner.status
    print(f"[check] GUI branch ran to completion in {frames} app frames", flush=True)
    print(runner.summary(), flush=True)
    code = 0 if (status is not None and status.success) else 1
    print(f"[check] {'PASS' if code == 0 else 'FAIL'}: mission success = "
          f"{status.success if status else None}", flush=True)

sys.stdout.flush()
simulation_app.close()
sys.exit(code)
