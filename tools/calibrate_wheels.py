import numpy as np
from src.sim.environment import Environment

# Ground truth is used here for CALIBRATION only, never in the controller.
env = Environment(seed=0) #builds the environment


#helper to keep an angle between -180 deg and +180 deg:
#   Needed to prevent code misreading changes in angles as sudden 300+ deg jumps.
def wrap(a): 
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---- Test 1: drive straight for 5 s -> wheel radius ----
env.reset() #puts robot at starting point (0,0) facing forward
l0, r0 = env.get_wheel_angles() #records starting wheel angles
p0 = env.get_ground_truth() #records starting true position (0 means at start)

for _ in range(250): #drives both wheels at 5 rad/s for 250 ticks (250 x 0.02 = 5 sec) --> same speed on both wheels drives it straight
    env.set_wheel_speeds(5.0, 5.0)
    env.step()

l1, r1 = env.get_wheel_angles() #record ending wheel angles
p1 = env.get_ground_truth() #record ending position

#Calculations to work out how far robot went:
dist = np.hypot(*(p1[:2] - p0[:2]))
dphi = ((l1 - l0) + (r1 - r0)) / 2
radius = dist / dphi

print(f"Commanded 5.00 rad/s, achieved {dphi / 5:.2f} rad/s")
print(f"Distance travelled: {dist:.3f} m")
print(f"Wheel radius r = {radius:.4f} m")

# ---- Test 2: spin on the spot for 5 s -> track width ----
env.reset() #back to start
l0, r0 = env.get_wheel_angles()
yaw_prev = env.get_ground_truth()[2]
yaw_total = 0.0

for _ in range(250): #Spins left wheel backwards, right wheel forwards = robot spins anticlockwise for 5 sec
    env.set_wheel_speeds(-3.0, 3.0)
    env.step()
    yaw = env.get_ground_truth()[2] #Every new tick reads new yaw, works out how much it turned since last tick, remebers this yaw for next time.
    yaw_total += wrap(yaw - yaw_prev)
    yaw_prev = yaw

#Track width formula:
l1, r1 = env.get_wheel_angles()
track = radius * ((r1 - r0) - (l1 - l0)) / yaw_total

print(f"Robot turned {np.degrees(yaw_total):.1f} degrees")
print(f"Track width b = {track:.4f} m")