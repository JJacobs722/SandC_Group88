"""Calibrate the real wheels with a tape measure (drive 1 m, spin 3 turns).
  python -m tools.calibrate_wheels_real
"""
import math

from src.hw.real_robot import RealRobot
from src.control.lqr_tracker import to_wheels, R_WHEEL, TRACK

env = RealRobot()
try:
    # ---- test 1: straight ----
    input("Test 1: mark the robot's front edge with tape, clear 1.5 m ahead, press Enter...")
    l0, r0 = env.get_wheel_angles()
    while True:
        l, r = env.get_wheel_angles()
        if R_WHEEL * ((l - l0) + (r - r0)) / 2 >= 1.0:      # wheels (spec radius) say 1 m
            break
        env.set_wheel_speeds(*to_wheels(0.08, 0.0))
        env.step()
    env.stop()
    l1, r1 = env.get_wheel_angles()
    dphi = ((l1 - l0) + (r1 - r0)) / 2                       # rad the wheels turned, on average
    real = float(input("How far did it REALLY go, in metres (e.g. 0.97)? "))
    radius = real / dphi

    # ---- test 2: spin ----
    input("\nTest 2: line the robot up with a tape line on the floor, press Enter to spin 3 turns...")
    l0, r0 = env.get_wheel_angles()
    while True:
        l, r = env.get_wheel_angles()
        if R_WHEEL * ((r - r0) - (l - l0)) / TRACK >= 3 * 2 * math.pi:   # wheels (spec) say 3 turns
            break
        env.set_wheel_speeds(*to_wheels(0.0, 0.8))
        env.step()
    env.stop()
    l1, r1 = env.get_wheel_angles()
    turns = float(input("How many turns did it REALLY do (e.g. 2.9, count past the tape line)? "))
    base = radius * ((r1 - r0) - (l1 - l0)) / (turns * 2 * math.pi)

    print(f"\nWHEEL_RADIUS = {radius:.4f}   (spec 0.033, sim effective 0.0281)")
    print(f"WHEEL_BASE   = {base:.4f}   (spec 0.287, sim effective 0.351)")
    print("Paste these into src/hw/hw_config.py. Comparing real vs sim numbers is good HD evidence.")
finally:
    env.close()
