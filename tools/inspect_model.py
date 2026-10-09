import sys
import mujoco

m = mujoco.MjModel.from_xml_path(sys.argv[1])

print("Joints:")
for i in range(m.njnt):
    print("  ", m.joint(i).name)
print("Actuators:")
for i in range(m.nu):
    print("  ", m.actuator(i).name)
print("Sensors:")
for i in range(m.nsensor):
    print("  ", m.sensor(i).name)
print("Cameras:")
for i in range(m.ncam):
    print("  ", m.camera(i).name)
print("Timestep:", m.opt.timestep)