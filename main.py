import time
from pathlib import Path
import mujoco
import mujoco.viewer

ROOT = Path(__file__).parent
MODEL = ROOT / "assets" / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml"

m = mujoco.MjModel.from_xml_path(str(MODEL))
d = mujoco.MjData(m)

print("Control range:", m.actuator_ctrlrange)

step = 0
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        d.ctrl[:] = 5.0     # bigger command to both wheels
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(m.opt.timestep)

        step += 1
        if step % 500 == 0:  # about once per simulated second
            x, y = d.qpos[0], d.qpos[1]
            print(f"t = {d.time:.1f} s   x = {x:.3f}   y = {y:.3f}")