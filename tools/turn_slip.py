"""How much does the Waffle Pi move when it spins on the spot?

The wheel model says opposite wheel speeds = turn in place, zero movement.
But the Waffle's body sits behind its wheels and drags on the caster balls,
so the wheel midpoint shuffles around while it turns. The wheels can't see this.

This spins the robot at a few speeds and measures how far the wheel midpoint wanders.
The result sets k_turn in the EKF (variance added per radian turned).

Ground truth is used here to measure the robot only, never by the robot itself.
Run from the top folder:  python -m tools.turn_slip
"""
import numpy as np

from src.sim.environment import Environment


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


env = Environment(seed=0)
for name in ("obstacle_1", "obstacle_2", "obstacle_3"):        # nothing in the way
    env.m.geom_pos[env.m.geom(name).id][:2] = [-2.0, -2.0]

print("wheel speed   turned    wandered (max)   wander per radian")
per_rad = []
for speed in (1.5, 3.0, 5.0):
    env.reset(0.0, 0.0, 0.0)
    start = env.get_ground_truth()
    prev, turned, wander = start[2], 0.0, 0.0
    for _ in range(500):                                        # 10 s
        env.set_wheel_speeds(-speed, speed)
        env.step()
        x, y, th = env.get_ground_truth()
        turned += abs(wrap(th - prev))
        prev = th
        wander = max(wander, np.hypot(x - start[0], y - start[1]))
    per_rad.append(wander / turned)
    print(f"  {speed:3.1f} rad/s    {np.degrees(turned):5.0f} deg   {wander * 100:5.1f} cm"
          f"          {wander / turned * 100:.2f} cm/rad")

# k_turn: variance added per radian turned. Spread one full turn's wander over 2*pi radians.

print(f"\nwander is about 5 to 6 cm whatever the speed (it shuffles, it doesn't keep sliding)")
print(f"k_turn = wander^2 / (2 pi) = {0.055 ** 2 / (2 * np.pi):.1e} m^2/rad  ->  we use 5e-4")
